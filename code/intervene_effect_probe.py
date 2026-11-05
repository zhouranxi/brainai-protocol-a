#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务四效应量探测（ADR-0016 候选 B / ADR-0006 D3）：
表征几何的推理期定向干预 → OOD 性能的**因果**变化。

问题（ADR-0016 原文）：
  R1 观察到「GLUE dimension 在 ID 受控下正相关预测 OOD」。
  这是相关性。本脚本问：**操纵 D_eff 能否因果地改变 OOD？**

设计：
  1. 固定 ID 准确率的 8 个 resnet18（id_acc 0.851–0.884，覆盖区间）
  2. 每个模型：抽 ID test / CIFAR-100 train+test / CIFAR-10C 特征（penultimate, 512d）
  3. 在 ID 特征上拟合干预算子（只用 ID 数据——预后性设定）：
     - flatten(α) : X' = X + α·σ_w·ε，ε~N(0,I)。放大所有方向方差 → 类流形
                    特征谱平坦化 → D_eff 升
     - sharpen(k) : X' = μ + V_k V_kᵀ (X−μ)，V_k = 类内协方差前 k 个主轴。
                    去掉尾部方向 → 90% 方差所需成分数减少 → D_eff 降
     - placebo(α) : X' = X + α·σ_w·u，u 为固定随机单位向量。范数匹配的安慰剂
  4. 同一算子作用于 OOD 特征（算子 ID-only 定义 → 迁移到 OOD 是合法的
     因果探针设定），测：ID acc（fc 头）、c100 linear probe、c10c acc、
     扰动后 ID 特征上的 GLUE 三指标（操纵检查）
  5. 效应量 = 每模型 d(OOD)/d(D_eff) 斜率（含基线共 5 点/族），
     对照 48 模型观察斜率（r≈+0.92）

判读：
  - flatten 族：ΔD_eff>0 且 ΔOOD 随动 → 因果链成立
  - sharpen 族：ΔD_eff<0；若 ΔOOD 反而升 → 观察相关非因果（与测量论文
    「配方决定指标」结论互恰）
  - placebo：D_eff 与 OOD 都应基本不动
  - 任何族 ΔD_eff 无单调响应 → 操纵失败，该族作废（不作因果声明）

输出：runs/intervene_effect_probe.json（逐行）+ 同名 summary
续跑：已完成的模型自动跳过。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r1_id_ood import (build_model, extract_features, linear_probe,  # noqa: E402
                       make_loader, subsample_per_class)
from glue_impl import glue_analysis, downsample_manifolds  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "results").is_dir() else HERE.parent

# 效应量探测用 8 模型：id_acc 0.851–0.884， quantile 挑选覆盖区间
DEFAULT_MODELS = [
    "resnet18_lr0.125_wd2.5e-05_s1",   # 0.8508
    "resnet18_lr1.0_wd2.5e-05_s0",     # 0.8630
    "resnet18_lr1.0_wd5e-05_s0",       # 0.8667
    "resnet18_lr0.125_wd0.0001_s0",    # 0.8722
    "resnet18_lr0.5_wd5e-05_s2",       # 0.8743
    "resnet18_lr0.5_wd0.0002_s2",      # 0.8779
    "resnet18_lr0.5_wd0.0001_s1",      # 0.8811
    "resnet18_lr0.125_wd0.0002_s2",    # 0.8838
]

# (kind, param)；param=None 即基线
CONDITIONS = (
    [("base", None)]
    + [("flatten", a) for a in (0.5, 1.0, 2.0, 4.0)]
    + [("sharpen", k) for k in (0.75, 0.5, 0.25, 0.1)]
    + [("placebo", a) for a in (1.0, 2.0, 4.0)]
)


# --------------------------------------------------------------------------- #
def fit_operator(X, y, num_classes=10):
    """在 ID 特征上拟合干预算子：类内协方差谱 + 全局均值。ID-only。"""
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[1]
    mu = X.mean(0)
    Sw = np.zeros((d, d), dtype=np.float64)
    for c in range(num_classes):
        Xc = X[y == c]
        Xcc = Xc - Xc.mean(0)
        Sw += Xcc.T @ Xcc
    Sw /= max(len(X), 1)
    lam, V = np.linalg.eigh(Sw)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    return {"mu": mu, "lam": lam, "V": V,
            "sigma": float(np.sqrt(max(lam.mean(), 1e-12)))}


def transform(X, op, kind, param, u_placebo):
    """把算子作用到任意特征矩阵（类无关的固定映射）。"""
    if kind == "base":
        return np.asarray(X)
    if kind == "flatten":
        # 确定性种子（跨进程可复现，不用 hash()——PYTHONHASHSEED 会随机化）
        rng = np.random.default_rng(10_000 + int(round(param * 10)))
        return np.asarray(X, dtype=np.float64) \
            + param * op["sigma"] * rng.standard_normal(np.asarray(X).shape)
    if kind == "sharpen":
        d = op["V"].shape[0]
        k = max(1, int(round(param * d)))
        Vk = op["V"][:, :k]
        return op["mu"] + (np.asarray(X, dtype=np.float64) - op["mu"]) @ Vk @ Vk.T
    if kind == "placebo":
        return np.asarray(X, dtype=np.float64) + param * op["sigma"] * u_placebo
    raise ValueError(kind)


@torch.no_grad()
def head_acc(fc, X, y, device):
    """用原分类头评估（扰动后）特征。"""
    xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    pred = fc(xt).argmax(1).cpu().numpy()
    return float((pred == np.asarray(y)).mean())


def glue_on(Xg, yg, m_per_class, n_points, n_t, seed):
    mans = [Xg[c * m_per_class:(c + 1) * m_per_class].T for c in range(10)]
    sub = downsample_manifolds(mans, n_points=n_points, seed=seed)
    return glue_analysis(sub, n_t=n_t, seed=seed)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir",
                    default="/mnt/mydisk/xinguan-brainai/runs/ckpt_grid")
    ap.add_argument("--data",
                    default=os.environ.get("BRAINAI_DATA_ROOT",
                                           "/mnt/mydisk/xinguan-brainai/data"))
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--probe-epochs", type=int, default=40)
    ap.add_argument("--m-per-class", type=int, default=500)
    ap.add_argument("--n-points", type=int, default=50)
    ap.add_argument("--n-t", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="调试：只跑前 N 个条件")
    args = ap.parse_args()

    device = args.device
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / "intervene_effect_probe.json"

    conditions = CONDITIONS[:args.limit] if args.limit else CONDITIONS
    models = [m for m in args.models.split(",") if m]

    rows, done = [], set()
    if fp.exists():
        try:
            rows = json.load(open(fp))["rows"]
            done = {r["model"] for r in rows
                    if r.get("_complete", True) and r["kind"] == "base"}
        except Exception:
            rows = []
    print(f"models={len(models)} conditions={len(conditions)} "
          f"done={sorted(done)}", flush=True)

    # ---- 数据加载器（构造一次，循环复用） ----
    data = Path(args.data)
    loaders = {
        "id": make_loader(str(data / "cifar_full" / "test"), 256),
        "c100_tr": make_loader(str(data / "cifar100" / "train"), 256),
        "c100_te": make_loader(str(data / "cifar100" / "test"), 256),
    }
    merged = data / "cifar10c_merged"
    if merged.is_dir():
        loaders["c10c"] = make_loader(str(merged), 256)

    for mi, name in enumerate(models):
        if name in done:
            print(f"[skip] {name} already done", flush=True)
            continue
        t0 = time.time()
        ck = Path(args.ckpt_dir) / f"{name}.pt"
        if not ck.exists():
            print(f"[missing] {ck}", flush=True)
            continue
        model = build_model().to(device)
        try:
            sd = torch.load(ck, map_location="cpu")
        except Exception:
            sd = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        model.eval()
        fc = model.fc

        # ---- 特征抽一次 ----
        Xid, _, yid = extract_features(model, loaders["id"], device)
        Xc_tr, _, yc_tr = extract_features(model, loaders["c100_tr"], device)
        Xc_te, _, yc_te = extract_features(model, loaders["c100_te"], device)
        Xk, _, yk = (extract_features(model, loaders["c10c"], device)
                     if "c10c" in loaders else (None, None, None))

        # ---- 算子拟合（ID-only）+ 固定安慰剂方向 ----
        op = fit_operator(Xid, yid)
        u = np.random.default_rng(1234).standard_normal(Xid.shape[1])
        u /= np.linalg.norm(u)

        # GLUE 子样本索引（固定 → 各条件配对同一批图）
        Xsub, ysub = subsample_per_class(Xid, yid, args.m_per_class, 10,
                                         seed=args.seed)

        for kind, param in conditions:
            tc = time.time()
            Xid_p = transform(Xid, op, kind, param, u)
            Xsub_p = transform(Xsub, op, kind, param, u)
            Xc_tr_p = transform(Xc_tr, op, kind, param, u)
            Xc_te_p = transform(Xc_te, op, kind, param, u)

            g = glue_on(Xsub_p, ysub, args.m_per_class, args.n_points,
                        args.n_t, args.seed)
            rec = {
                "model": name, "kind": kind, "param": param,
                "glue_d": round(float(g["dimension"]), 4),
                "glue_r": round(float(g["radius"]), 4),
                "glue_u": round(float(g["utility"]), 4),
                "id_acc": round(head_acc(fc, Xid_p, yid, device), 4),
                "c100_probe": round(linear_probe(
                    torch.as_tensor(Xc_tr_p, dtype=torch.float32, device=device),
                    torch.as_tensor(yc_tr, dtype=torch.long, device=device),
                    torch.as_tensor(Xc_te_p, dtype=torch.float32, device=device),
                    torch.as_tensor(yc_te, dtype=torch.long, device=device),
                    num_classes=100, epochs=args.probe_epochs, seed=args.seed), 4),
                "c10c_acc": (round(head_acc(fc, transform(Xk, op, kind, param, u),
                                          yk, device), 4)
                             if Xk is not None else None),
                "lr": float(sd.get("lr", np.nan)),
                "wd": float(sd.get("wd", np.nan)),
                "seed_ckpt": int(sd.get("seed", -1)),
                "secs": round(time.time() - tc, 1),
                "_complete": True,
            }
            rows.append(rec)
            print(f"[{mi+1}/{len(models)}] {name} {kind}={param}: "
                  f"D={rec['glue_d']:.3f} id={rec['id_acc']:.4f} "
                  f"c100={rec['c100_probe']:.4f} c10c={rec['c10c_acc']} "
                  f"({rec['secs']}s)", flush=True)
            fp.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        print(f"[{mi+1}/{len(models)}] {name} done in {time.time()-t0:.0f}s",
              flush=True)

    summarize(rows, outdir)
    return 0


def summarize(rows, outdir):
    """效应量汇总：每模型 × 族 的 d(OOD)/d(D_eff) 斜率 + 操纵检查。"""
    from scipy.stats import pearsonr
    models = sorted({r["model"] for r in rows})
    out = {"per_model": [], "pooled": {}}
    for fam in ("flatten", "sharpen", "placebo"):
        slopes_oob, slopes_c10c, manips = [], [], []
        for m in models:
            pts = [r for r in rows if r["model"] == m and r["kind"] == fam]
            base = [r for r in rows if r["model"] == m and r["kind"] == "base"]
            if not base:
                continue
            pts = pts + base
            if len(pts) < 3:            # 至少 base + 2 个强度点才有斜率可言
                continue
            D = np.array([r["glue_d"] for r in pts])
            Y = np.array([r["c100_probe"] for r in pts])
            K = np.array([r["c10c_acc"] for r in pts]) \
                if all(r["c10c_acc"] is not None for r in pts) else None
            sl = float(np.polyfit(D, Y, 1)[0])
            slopes_oob.append(sl)
            if K is not None and len(K) >= 2:
                slopes_c10c.append(float(np.polyfit(D, K, 1)[0]))
            manips.append(float(pearsonr(
                D, np.arange(len(D)))[0]))  # D 对条件序的单调响应
        if slopes_oob:
            out["pooled"][fam] = {
                "slope_c100_per_model": [round(s, 4) for s in slopes_oob],
                "slope_c100_median": round(float(np.median(slopes_oob)), 4),
                "slope_c10c_median": (round(float(np.median(slopes_c10c)), 4)
                                      if slopes_c10c else None),
                "manipulation_check_median_r": (round(float(np.median(manips)), 3)
                                                if manips else None),
            }
    fp = Path(outdir) / "intervene_effect_probe_summary.json"
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["pooled"], indent=1), flush=True)
    print(f"[saved] {fp}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
