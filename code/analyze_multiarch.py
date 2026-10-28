#!/usr/bin/env python3
"""三架构（resnet18 / shufflenet_v2 / regnet）合并的跨架构 Protocol A。

混杂源 = 真正的架构维度（不再用 lr 族替代）。
数据：ind_validation_results.json (resnet18, 48) + mod_arch_results.json (新架构)。
OOD：c100 与 tinyimg 两个类级偏移。
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


def safe_corr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 4 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan"), float("nan"), float("nan")
    r, p = pearsonr(x, y)
    rho, _ = spearmanr(x, y)
    return float(r), float(p), float(rho), float(pr := rho)


def group_residual(v, groups):
    v = np.asarray(v, float)
    uniq = sorted(set(groups))
    if len(uniq) < 2:
        return v - v.mean()
    D = np.column_stack([(np.asarray(groups) == g).astype(float) for g in uniq[1:]])
    A = np.column_stack([np.ones(len(v)), D])
    beta, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ beta


def _find_data_file(name):
    """Locate a shipped data file under any of the layouts we publish."""
    for cand in (ROOT / "data" / "self_trained" / name,
                 ROOT / "results" / name,
                 ROOT / name):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{name} not found under {ROOT}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ind", default=None, help="path to ind_validation_results.json")
    ap.add_argument("--mod", default=None, help="path to mod_arch_results.json")
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--band", type=float, default=2.0)
    ap.add_argument("--min-acc", type=float, default=0.3,
                    help="剔除发散模型（id_acc 低于此值）")
    args = ap.parse_args()
    if not args.ind:
        args.ind = str(_find_data_file("ind_validation_results.json"))
    if not args.mod:
        args.mod = str(_find_data_file("mod_arch_results.json"))
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    if os.path.exists(args.ind):
        for r in json.load(open(args.ind))["rows"]:
            r["arch"] = "resnet18"
            rows.append(r)
    if os.path.exists(args.mod):
        for r in json.load(open(args.mod))["rows"]:
            rows.append(r)

    # 剔除发散模型
    n0 = len(rows)
    rows = [r for r in rows if r.get("id_acc", 0) >= args.min_acc]
    L = []
    def w(s=""):
        L.append(s)
        print(s, flush=True)

    w("# 三架构合并的跨架构 Protocol A")
    w("")
    w(f"模型：{len(rows)}（剔除发散后；原始 {n0}）")
    archs = sorted({r["arch"] for r in rows})
    for a in archs:
        w(f"  - {a}: {sum(1 for r in rows if r['arch']==a)}")
    w("")
    w("混杂源 = 架构（真正的跨架构维度）。OOD = c100 / tinyimg 两个类级偏移。")
    w("")

    MK = sorted({k for r in rows for k in r if k.startswith("id_") and k != "id_acc"})

    id_pct = np.array([r["id_acc"] for r in rows]) * 100
    arch = np.array([r["arch"] for r in rows])

    for ood_key, ood_name in [("c100_probe_acc", "CIFAR-100"),
                              ("tinyimg_probe_acc", "Tiny-ImageNet")]:
        have = [r for r in rows if ood_key in r]
        rows_local = have
        w(f"## OOD = {ood_name}（n={len(have)}）")
        ood = np.array([r[ood_key] for r in have]) * 100
        acc = np.array([r["id_acc"] for r in have]) * 100
        ga = np.array([r["arch"] for r in have])

        # 带宽内子集
        m = (acc >= acc.mean() - args.band) & (acc <= acc.mean() + args.band)
        w(f"ID 受控子集（band=±{args.band:.0f}pp）：n={m.sum()}，架构构成 "
          + ", ".join(f"{a}:{(ga[m]==a).sum()}" for a in sorted(set(ga[m]))))
        if m.sum() < 12 or len(set(ga[m])) < 2:
            w("子集不足，跳过。\n")
            continue
        s_acc, s_ood, s_ga = acc[m], ood[m], ga[m]
        w(f"ID 跨度 {s_acc.max()-s_acc.min():.2f}pp；"
          f"OOD 跨度 {s_ood.max()-s_ood.min():.2f}pp"
          f"（全体 {ood.max()-ood.min():.2f}，"
          f"保留 {(s_ood.max()-s_ood.min())/(ood.max()-ood.min())*100:.0f}%）")
        w("")
        ood_res = group_residual(s_ood, s_ga)
        w("| 预测器 | 原始 r | 残差化 r | p | Spearman ρ | 逐架构符号 | 一致? |")
        w("|---|---|---|---|---|---|---|")
        for mk in MK:
            xs = np.array([r[mk] for r in have])[m]
            raw_r, _, _, _ = safe_corr(xs, s_ood)
            rx = group_residual(xs, s_ga)
            r_, p_, rho_, _ = safe_corr(rx, ood_res)
            cells, signs = [], []
            for a in sorted(set(s_ga)):
                sel = s_ga == a
                if sel.sum() < 5:
                    cells.append("n<5")
                    continue
                rr, _, _, _ = safe_corr(xs[sel], s_ood[sel])
                cells.append(f"{rr:+.2f}")
                if not np.isnan(rr):
                    signs.append(np.sign(rr))
            cons = "✓" if signs and len(set(signs)) == 1 else "✗"
            w(f"| `{mk}` | {raw_r:+.3f} | {r_:+.3f} | {p_:.4f} | {rho_:+.3f} | "
              f"{'/'.join(cells)} | {cons} |")
        raw_r, _, _, _ = safe_corr(s_acc, s_ood)
        rx = group_residual(s_acc, s_ga)
        r_, p_, rho_, _ = safe_corr(rx, ood_res)
        cells, signs = [], []
        for a in sorted(set(s_ga)):
            sel = s_ga == a
            if sel.sum() < 5:
                continue
            rr, _, _, _ = safe_corr(s_acc[sel], s_ood[sel])
            if not np.isnan(rr):
                signs.append(np.sign(rr))
            cells.append(f"{rr:+.2f}")
        w(f"| **`id_acc`（基线）** | {raw_r:+.3f} | **{r_:+.3f}** | {p_:.4f} | "
          f"{rho_:+.3f} | {'/'.join(cells)} | "
          f"{'✓' if signs and len(set(signs))==1 else '✗'} |")
        w("")

        # ---- 组内协议：每个架构用自身均值做受控子集 ----
        w(f"### 组内协议（每架构独立，band=±{args.band:.0f}pp 于各自均值）")
        w("")
        w("| 架构 | n | `id_pr_perclass` 残差化 r | `id_acc` 残差化 r | 基线符号 |")
        w("|---|---|---|---|---|")
        for a in sorted(set(ga)):
            arch_rows = [r for r in have if r["arch"] == a]
            aa = np.array([r["id_acc"] for r in arch_rows]) * 100
            mm = np.abs(aa - aa.mean()) <= args.band
            if mm.sum() < 8:
                w(f"| {a} | {mm.sum()} | 样本不足 | — | — |")
                continue
            sub_rows = [arch_rows[i] for i in np.where(mm)[0]]
            s2_acc = np.array([r["id_acc"] for r in sub_rows]) * 100
            s2_ood = np.array([r[ood_key] for r in sub_rows]) * 100
            lr2 = np.array([r["lr"] for r in sub_rows])
            o2 = group_residual(s2_ood, lr2)
            x2 = group_residual(np.array([r["id_pr_perclass"] for r in sub_rows]), lr2)
            r2, p2, rho2, _ = safe_corr(x2, o2)
            x3 = group_residual(s2_acc, lr2)
            r3, p3, rho3, _ = safe_corr(x3, o2)
            w(f"| {a} | {len(sub_rows)} | {r2:+.3f} (p={p2:.4f}) | {r3:+.3f} (p={p3:.4f}) | "
              f"{'正' if r3>0 else '负'} |")
        w("")

    fp = os.path.join(args.outdir, "multiarch_report.md")
    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[saved] {fp}")


if __name__ == "__main__":
    main()
