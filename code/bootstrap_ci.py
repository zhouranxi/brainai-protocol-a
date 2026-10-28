#!/usr/bin/env python3
"""分层 bootstrap 置信区间：为 Protocol A 的关键相关系数补 CI。

设计要点（这是与"只对最后一步相关性做 bootstrap"的区别）：
  每个 bootstrap 样本内**重新执行完整协议**：
    1. 按分组变量（架构 / lr 族）**分层重采样**（有放回，层内等量）
       —— 保持各层比例，避免 bootstrap 样本的架构构成漂移
    2. 在该样本上**重新筛选 ID 受控子集**（band 以样本内 mean 为中心）
    3. **重新做架构残差化**（残差化矩阵随样本变化）
    4. 计算残差化 Pearson 与 Spearman
  → CI 反映的是**整个流程**的不确定性。

输出：
  B1  224 模型主数据：稳健标记 + 基线的 95% CI
  B2  80 模型 corruption 数据（gaussian_noise 场景）
  B3  自建 48 组
  B4  稳健判定的复现率（bootstrap 中判据通过的比例）
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
import pandas as pd
from scipy.stats import pearsonr, spearmanr

GLUE = ["dimension_id", "radius_id", "utility_id", "center_alignment_id",
        "axis_alignment_id", "center_axis_alignment_id", "capacity_id"]
GENERIC = ["participation_ratio_id", "numerical_rank_id", "nc1_id",
           "mean_distance_id", "mean_distance_avg_id", "mean_angle_id",
           "mean_covariance_id", "sparsity_id"]


def safe_corr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 4 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan")
    r, _ = pearsonr(x, y)
    rho, _ = spearmanr(x, y)
    return float(r), float(rho)


def arch_residual(v, groups):
    v = np.asarray(v, float)
    uniq = sorted(set(groups))
    if len(uniq) < 2:
        return v - v.mean()
    D = np.column_stack([(np.asarray(groups) == g).astype(float) for g in uniq[1:]])
    A = np.column_stack([np.ones(len(v)), D])
    beta, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ beta


def stratified_indices(groups, rng):
    """分层有放回重采样：每层内等量抽回，返回索引数组。"""
    groups = np.asarray(groups)
    out = []
    for g in sorted(set(groups)):
        idx = np.where(groups == g)[0]
        out.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(out)


def protocol_once(id_acc, ood_acc, groups, marker_vals, band, min_n=20):
    """执行一次完整协议：ID 受控 → 残差化 → 双相关性。"""
    id_acc = np.asarray(id_acc, float)
    ood_acc = np.asarray(ood_acc, float)
    m = (id_acc >= id_acc.mean() - band) & (id_acc <= id_acc.mean() + band)
    if m.sum() < min_n or len(set(np.asarray(groups)[m])) < 2:
        return None
    s_id, s_ood, s_grp = id_acc[m], ood_acc[m], np.asarray(groups)[m]
    ood_res = arch_residual(s_ood, s_grp)
    out = {}
    for name, x in marker_vals.items():
        xx = np.asarray(x, float)[m]
        rx = arch_residual(xx, s_grp)
        r, rho = safe_corr(rx, ood_res)
        out[name] = (r, rho)
    rb, rhob = safe_corr(arch_residual(s_id, s_grp), ood_res)
    out["acc_test"] = (rb, rhob)
    return out


def bootstrap_ci(id_acc, ood_acc, groups, marker_vals, band, n_boot, seed=0,
                 min_n=20):
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    stats = {}
    keys = list(marker_vals.keys()) + ["acc_test"]
    for k in keys:
        stats[k] = {"r": [], "rho": [], "robust": []}
    for _ in range(n_boot):
        idx = stratified_indices(groups, rng)
        res = protocol_once(np.asarray(id_acc, float)[idx],
                            np.asarray(ood_acc, float)[idx],
                            groups[idx],
                            {k: np.asarray(v, float)[idx] for k, v in marker_vals.items()},
                            band, min_n)
        if res is None:
            continue
        for k, (r, rho) in res.items():
            stats[k]["r"].append(r)
            stats[k]["rho"].append(rho)
            # 稳健判定：双指标同向且 |rho|>=0.5
            if not np.isnan(r) and not np.isnan(rho):
                stats[k]["robust"].append(
                    bool(np.sign(r) == np.sign(rho) and abs(rho) >= 0.5))
            else:
                stats[k]["robust"].append(False)
    return stats


def summarize(stats, alpha=0.05):
    out = {}
    for k, d in stats.items():
        r = np.array([x for x in d["r"] if not np.isnan(x)])
        rho = np.array([x for x in d["rho"] if not np.isnan(x)])
        rob = np.array(d["robust"], dtype=bool)
        if len(r) < 10:
            out[k] = None
            continue
        out[k] = {
            "r_mean": float(r.mean()),
            "r_ci": [float(np.percentile(r, 100 * alpha / 2)),
                     float(np.percentile(r, 100 * (1 - alpha / 2)))],
            "rho_mean": float(rho.mean()),
            "rho_ci": [float(np.percentile(rho, 100 * alpha / 2)),
                       float(np.percentile(rho, 100 * (1 - alpha / 2)))],
            "r_excl0": bool(np.percentile(r, 100 * alpha / 2) > 0 or
                            np.percentile(r, 100 * (1 - alpha / 2)) < 0),
            "rho_excl0": bool(np.percentile(rho, 100 * alpha / 2) > 0 or
                              np.percentile(rho, 100 * (1 - alpha / 2)) < 0),
            "same_sign": bool(np.sign(r.mean()) == np.sign(rho.mean())),
            "robust_rate": float(rob.mean()),
            "n_valid": int(len(r)),
        }
    return out


def fmt_ci(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--band-fig3", type=float, default=1.0)
    ap.add_argument("--band-80", type=float, default=2.0)
    ap.add_argument("--band-own", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    L = []
    def w(s=""):
        L.append(s)
        print(s, flush=True)

    w("# 分层 Bootstrap 置信区间（Protocol A 全流程）")
    w("")
    w(f"n_boot = {args.n_boot}，分层单位 = 分组变量（架构 / lr 族），"
      "每个 bootstrap 样本内重新执行 ID 受控筛选 + 残差化 + 双相关性。")
    w("CI 为 percentile 法 95%。`排除0` = CI 不含 0；"
      "`同号` = Pearson 与 Spearman 均值同号；`稳健率` = bootstrap 中判据通过比例。")
    w("")

    all_out = {}

    # ================= B1: 224 模型 =================
    d = pd.read_pickle(ROOT / "data" / "official" / "df_fig3.pkl").reset_index(drop=True)
    MARKERS = [m for m in GLUE + GENERIC if m in d.columns]
    marker_vals = {m: d[m].values.astype(float) for m in MARKERS}
    w(f"## B1 官方 224 模型（band = ±{args.band_fig3:.0f}pp）")
    w("")
    stats = bootstrap_ci(d["acc_test"].values, d["acc_lp_ood"].values,
                         d["arch_name"].values, marker_vals,
                         args.band_fig3, args.n_boot, args.seed)
    summ = summarize(stats)
    all_out["B1_fig3_224"] = summ
    w("| 预测器 | 残差化 r | r 的 95% CI | 排除0 | Spearman ρ | ρ 的 95% CI | 排除0 | 同号 | 稳健率 |")
    w("|---|---|---|---|---|---|---|---|---|")
    order = [m for m in MARKERS if summ.get(m)] + (["acc_test"] if summ.get("acc_test") else [])
    order.sort(key=lambda k: -abs(summ[k]["r_mean"]))
    for k in order:
        s = summ[k]
        w(f"| `{'acc_test（基线）' if k == 'acc_test' else k}` | "
          f"{s['r_mean']:+.3f} | {fmt_ci(s['r_ci'])} | {'✓' if s['r_excl0'] else '✗'} | "
          f"{s['rho_mean']:+.3f} | {fmt_ci(s['rho_ci'])} | {'✓' if s['rho_excl0'] else '✗'} | "
          f"{'✓' if s['same_sign'] else '✗'} | {s['robust_rate']*100:.0f}% |")
    w("")

    # ================= B2: 80 模型 =================
    d2 = pd.read_pickle(ROOT / "data" / "official" / "df_fig15-20.pkl")
    d2 = d2["gaussian_noise"].reset_index(drop=True)
    MARKERS2 = [m for m in GLUE + GENERIC if m in d2.columns]
    marker_vals2 = {m: d2[m].values.astype(float) for m in MARKERS2}
    w(f"## B2 官方 80 模型 corruption 数据（gaussian_noise，band = ±{args.band_80:.0f}pp）")
    w("")
    stats2 = bootstrap_ci(d2["acc_test"].values, d2["acc_lp_ood"].values,
                          d2["arch_name"].values, marker_vals2,
                          args.band_80, args.n_boot, args.seed)
    summ2 = summarize(stats2)
    all_out["B2_official_80"] = summ2
    w("| 预测器 | 残差化 r | r 的 95% CI | 排除0 | Spearman ρ | ρ 的 95% CI | 排除0 | 同号 | 稳健率 |")
    w("|---|---|---|---|---|---|---|---|---|")
    order2 = [m for m in MARKERS2 if summ2.get(m)] + (["acc_test"] if summ2.get("acc_test") else [])
    order2.sort(key=lambda k: -abs(summ2[k]["r_mean"]))
    for k in order2[:10]:
        s = summ2[k]
        w(f"| `{'acc_test（基线）' if k == 'acc_test' else k}` | "
          f"{s['r_mean']:+.3f} | {fmt_ci(s['r_ci'])} | {'✓' if s['r_excl0'] else '✗'} | "
          f"{s['rho_mean']:+.3f} | {fmt_ci(s['rho_ci'])} | {'✓' if s['rho_excl0'] else '✗'} | "
          f"{'✓' if s['same_sign'] else '✗'} | {s['robust_rate']*100:.0f}% |")
    w("")
    w("（B2 只列 |r| 前 10；完整结果在 JSON。）")
    w("")

    # ================= B3: 自建 48 组 =================
    own = json.load(open(_find_data_file("ind_validation_results.json")))
    rows = own["rows"]
    id3 = np.array([r["id_acc"] * 100 for r in rows])       # 百分数
    ood3 = np.array([r["c100_probe_acc"] * 100 for r in rows])
    grp3 = np.array([float(r["lr"]) for r in rows])
    MK3 = [k for k in rows[0] if k.startswith("id_") and k != "id_acc"]
    mv3 = {k: np.array([r[k] for r in rows], float) for k in MK3}
    w(f"## B3 自建 48 组（band = ±{args.band_own:.0f}pp，分层单位 = lr 族）")
    w("")
    stats3 = bootstrap_ci(id3, ood3, grp3, mv3, args.band_own, args.n_boot, args.seed)
    summ3 = summarize(stats3)
    all_out["B3_own_48"] = summ3
    w("| 预测器 | 残差化 r | r 的 95% CI | 排除0 | Spearman ρ | ρ 的 95% CI | 排除0 | 同号 | 稳健率 |")
    w("|---|---|---|---|---|---|---|---|---|")
    order3 = [m for m in MK3 if summ3.get(m)] + (["acc_test"] if summ3.get("acc_test") else [])
    order3.sort(key=lambda k: -abs(summ3[k]["r_mean"]))
    for k in order3:
        s = summ3[k]
        nm = "id_acc（基线）" if k == "acc_test" else k
        w(f"| `{nm}` | {s['r_mean']:+.3f} | {fmt_ci(s['r_ci'])} | {'✓' if s['r_excl0'] else '✗'} | "
          f"{s['rho_mean']:+.3f} | {fmt_ci(s['rho_ci'])} | {'✓' if s['rho_excl0'] else '✗'} | "
          f"{'✓' if s['same_sign'] else '✗'} | {s['robust_rate']*100:.0f}% |")
    w("")

    # ================= 结论 =================
    w("## 结论")
    w("")
    def verdict(k, s, label):
        if s is None:
            return f"- **{label}**：样本不足"
        parts = []
        if s["r_excl0"] and s["rho_excl0"] and s["same_sign"]:
            parts.append(f"r CI {fmt_ci(s['r_ci'])} 与 ρ CI {fmt_ci(s['rho_ci'])} "
                         f"均排除 0 且同号 → **效应稳健**")
        elif not s["r_excl0"] or not s["rho_excl0"]:
            parts.append(f"CI 含 0（r {fmt_ci(s['r_ci'])}，ρ {fmt_ci(s['rho_ci'])}）"
                         f" → **不稳健**")
        else:
            parts.append(f"两 CI 均排除 0 但**方向不一致**（r {s['r_mean']:+.3f} vs "
                         f"ρ {s['rho_mean']:+.3f}）→ **判据不通过，不可作基线**")
        parts.append(f"bootstrap 稳健率 {s['robust_rate']*100:.0f}%")
        return f"- **{label}**：" + "；".join(parts)

    w("### 224 模型（主数据）")
    for k in ["dimension_id", "utility_id", "radius_id", "acc_test"]:
        if summ.get(k):
            w(verdict(k, summ[k], "acc_test（基线）" if k == "acc_test" else k))
    w("")
    w("### 80 模型（corruption 数据）")
    for k in ["dimension_id", "utility_id", "acc_test"]:
        if summ2.get(k):
            w(verdict(k, summ2[k], "acc_test（基线）" if k == "acc_test" else k))
    w("")
    w("### 自建 48 组（独立验证）")
    for k in ["id_pr_perclass", "id_numerical_rank", "acc_test"]:
        if summ3.get(k):
            w(verdict(k, summ3[k], "id_acc（基线）" if k == "acc_test" else k))
    w("")

    out_md = os.path.join(args.outdir, "bootstrap_ci_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    out_json = os.path.join(args.outdir, "bootstrap_ci_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"n_boot": args.n_boot, "seed": args.seed,
                   "bands": {"fig3": args.band_fig3, "official80": args.band_80,
                             "own": args.band_own},
                   "results": all_out}, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_md}\n[saved] {out_json}")


if __name__ == "__main__":
    main()
