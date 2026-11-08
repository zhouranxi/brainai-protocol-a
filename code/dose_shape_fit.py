#!/usr/bin/env python3
"""剂量扫描形态判定（ADR-0016 附录二十三预注册判据 R1/R2/R3）。

数据：5 点对数轴 λ ∈ {0.03, 0.1, 0.3, 1.0, 3.0}
  - λ=0.1/1.0：全量批 full_s0_whiten_lam{0.1,1.0}（锚点两配方）
  - λ={0.03, 0.3, 3.0}：剂量扫描 dose_{lo,hi}_lam* 新 run
  - base 对照：full_base_lr*.json 同配方读数

拟合（每配方分别 + 合并池）：
  线性   Δc = a·Δd
  幂律   Δc = a·Δd^b
  对数   Δc = a·log(1+Δd/b0)
AICc 选优；判形态需 ΔAICc ≥ 2。

R2：幂律胜出 ⇒ b 的 bootstrap 95% CI 完全 <1 判「强凹」。
R3：λ=3.0 功能保持（ID≥base−2pp）时是否进入平台。
"""
import json
import math
from pathlib import Path

import numpy as np

BASE = Path("/mnt/mydisk/xinguan-brainai/runs/v2_p4_forcing")

MODELS = {
    "lo": "resnet18_lr1.0_wd5e-05_s0",
    "hi": "resnet18_lr0.125_wd0.0001_s0",
}
LAMS = [0.03, 0.1, 0.3, 1.0, 3.0]

# base 读数
base_by_tag = {}
for fp in sorted(BASE.glob("full_base_lr*.json")):
    for r in json.load(open(fp))["rows"]:
        base_by_tag[r["model"]] = r


def load_point(tag, lam):
    """找 (model, whiten, lam) 的读数，返回 (d_delta, c_delta, id_keep)。"""
    model = MODELS[tag]
    b = base_by_tag[model]
    cands = []
    if lam in (0.1, 1.0):
        cands.append(BASE / f"full_s0_whiten_lam{lam}" / "train_forcing_p4.json")
    cands.append(BASE / f"dose_{tag}_lam{lam}" / "train_forcing_p4.json")
    for fp in cands:
        if not fp.exists():
            continue
        for r in json.load(open(fp))["rows"]:
            if (r["model"] == model and r["kind"] == "whiten"
                    and abs(r["lambda"] - lam) < 1e-9):
                de = r["d_ent_id"] - b["d_ent_id"]
                dc = r["c100_probe"] - b["c100_probe"]
                ik = r["id_probe_acc"] - b["id_probe_acc"]
                return de, dc, ik
    return None


def fit_stats(x, y, pred_fn, k):
    """残差平方和 → AICc（正态误差假设）。k = 参数数（含 σ）。"""
    yh = pred_fn(x)
    n = len(y)
    rss = float(np.sum((y - yh) ** 2))
    ll = -n / 2 * (math.log(2 * math.pi * rss / n) + 1)
    aic = 2 * k - 2 * ll
    if n - k - 1 <= 0:
        return aic, rss
    aicc = aic + 2 * k * (k + 1) / (n - k - 1)
    return aicc, rss


def fit_all(x, y):
    """三模型拟合，返回 dict。"""
    out = {}
    # 线性（过原点：Δc=0 当 Δd=0，物理约束）
    A = np.vstack([x]).T
    a_lin = float(np.linalg.lstsq(A, y, rcond=None)[0][0])
    out["linear"] = fit_stats(x, y, lambda xx: a_lin * xx, 2)
    out["linear_coef"] = a_lin
    # 幂律 Δc = a·Δd^b（log-log 线性回归起步，再解析）
    lx, ly = np.log(x), np.log(y)
    b_pow, loga = np.polyfit(lx, ly, 1)
    out["power"] = fit_stats(x, y, lambda xx: math.exp(loga) * xx ** b_pow, 3)
    out["power_coef"] = (math.exp(loga), b_pow)
    # 对数 Δc = a·log(1+Δd/b0)：b0 网格扫描
    best = None
    for b0 in np.geomspace(1.0, 200.0, 400):
        z = np.log1p(x / b0)
        a = float(np.sum(z * y) / np.sum(z * z))
        f = lambda xx, a=a, b0=b0: a * np.log1p(xx / b0)
        s = fit_stats(x, y, f, 3)
        if best is None or s[0] < best[0][0]:
            best = (s, a, b0)
    out["log"], out["log_coef"] = best[0], (best[1], best[2])
    return out


def main():
    pts = {}
    for tag in MODELS:
        rows = []
        for lam in LAMS:
            p = load_point(tag, lam)
            if p is None:
                print(f"[warn] {tag}@λ={lam} 读数缺失")
                continue
            rows.append((lam,) + p)
        pts[tag] = rows
        print(f"\n=== {tag}（{MODELS[tag]}）===")
        print(f"{'λ':>6s} {'Δd_ent':>9s} {'Δc100':>9s} {'id_keep':>8s}")
        for lam, de, dc, ik in rows:
            flag = " ←NEW" if lam in (0.03, 0.3, 3.0) else ""
            keep = "ok" if ik >= -0.02 else "FUNC_LOSS"
            print(f"{lam:6g} {de:+9.2f} {dc:+9.4f} {ik:+8.4f} {keep}{flag}")

    for tag, rows in pts.items():
        if len(rows) < 4:
            print(f"\n[{tag}] 点数不足（{len(rows)}），跳过拟合")
            continue
        x = np.array([r[1] for r in rows if r[3] >= -0.02], dtype=float)
        y = np.array([r[2] for r in rows if r[3] >= -0.02], dtype=float)
        if len(x) < 4:
            print(f"\n[{tag}] 功能保持点不足，跳过")
            continue
        print(f"\n--- {tag} 形态拟合（n={len(x)}，功能保持点）---")
        fits = fit_all(x, y)
        order = sorted(["linear", "power", "log"], key=lambda k: fits[k][0])
        for k in order:
            aicc, rss = fits[k]
            print(f"  {k:7s} AICc={aicc:8.2f}  rss={rss:.6f}")
        d = fits[order[1]][0] - fits[order[0]][0]
        if d >= 2:
            print(f"  ⇒ 形态判定：{order[0]}（ΔAICc={d:.2f} ≥ 2）")
        else:
            print(f"  ⇒ 5 点不足以分形态（ΔAICc={d:.2f} < 2）")
        if order[0] == "power":
            a, b = fits["power_coef"]
            print(f"     幂律指数 b = {b:.3f}（<1 ⇒ 强凹，R2）")
        if order[0] == "log":
            a, b0 = fits["log_coef"]
            print(f"     对数饱和尺度 b0 = {b0:.1f}")

    # 合并池（两配方 10 点）
    xs, ys = [], []
    for tag, rows in pts.items():
        for r in rows:
            if r[3] >= -0.02:
                xs.append(r[1]); ys.append(r[2])
    if len(xs) >= 6:
        x, y = np.array(xs), np.array(ys)
        print(f"\n--- 合并池（n={len(x)}）---")
        fits = fit_all(x, y)
        order = sorted(["linear", "power", "log"], key=lambda k: fits[k][0])
        for k in order:
            print(f"  {k:7s} AICc={fits[k][0]:8.2f}")
        d = fits[order[1]][0] - fits[order[0]][0]
        print(f"  ΔAICc={d:.2f} ⇒ " + (f"{order[0]}" if d >= 2 else "不分形态"))
        if order[0] == "power":
            a, b = fits["power_coef"]
            print(f"  幂律指数 b = {b:.3f}")

    # R3：λ=3 平台检验
    print("\n--- R3：λ=3.0 平台检验 ---")
    for tag in MODELS:
        p = load_point(tag, 3.0)
        p1 = load_point(tag, 1.0)
        if p and p1:
            slope3 = (p[1] - p1[1])
            if slope3 > 0 and p[2] >= -0.02:
                incr = (p[0] - p1[0]) / (p1[0] if p1[0] else 1)
                print(f"  {tag}: λ1→3 Δd {p1[0]:+.1f}→{p[0]:+.1f}, "
                      f"Δc {p1[1]:+.4f}→{p[1]:+.4f}, 增量斜率 "
                      f"{(p[1]-p1[1])/(p[0]-p1[0]):+.5f}")
    print("DONE")


if __name__ == "__main__":
    main()
