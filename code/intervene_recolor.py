#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务四第二步：无噪声几何算子（谱重着色）——分离「几何 vs 噪声」两条因果路径。

承接 intervene_effect_probe.py 的发现（runs/probe/）：
  - flatten（噪声注入）推高 D_eff ×10 ⇒ c100 单调降（8/8 反号）
  - sharpen（无噪声投影）在低 lr 模型上 D_eff ↓ ⇒ c100 同向 ↓（与观察同号）
  - 2×2 表缺最后一格：**无噪声的 D_eff 升**

本脚本补这格：recolor(γ) —— 类内谱重着色
  x' = μ + V · diag(λ_i^((γ-1)/2)) · Vᵀ (x − μ)
  其中 Sw = V diag(λ) Vᵀ 为 ID 类内协方差（ID-only 拟合，与第一步同协议）。
  γ=1 恒等；γ=0 全白化（类内各方向等方差 → D_eff 最大）；γ=2 双倍锐化。
  **确定性线性映射、可逆（eps 截断前）、零噪声。**

对照（安慰剂）：tot_recolor(γ) —— 同形式算子，但用**全局总协方差**的谱。
  差别：类内谱决定流形形状，总谱还编码类间结构——若效应特异于类内谱，
  tot_recolor 应显著更弱（谱特异性检查）。

每个条件测：扰动后 GLUE 三指标、id_fc_acc（原头）、id_probe_acc（10-way 探针，
重训 → 测信息保持而非头部兼容）、c100 probe、c10c acc。

判读（预注册，跑前写下）：
  H1（几何因果）：recolor γ↓（D_eff↑、零噪声）⇒ c100 单调降，且与 flatten 幅度相当
    → D_eff 本身携因果代价；观察相关仍是伴随标记
  H2（噪声解释）：recolor γ↓ ⇒ c100 不动或升
    → 第一步的反号效应来自噪声注入而非几何；观察相关可能部分因果
  判据：逐模型 d(c100)/d(D_eff) 斜率符号（recolor 族，8/8 同号才算干净）

输出：runs/probe2/intervene_recolor.json + summary。断点续跑安全。
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

DEFAULT_MODELS = [
    "resnet18_lr0.125_wd2.5e-05_s1",
    "resnet18_lr1.0_wd2.5e-05_s0",
    "resnet18_lr1.0_wd5e-05_s0",
    "resnet18_lr0.125_wd0.0001_s0",
    "resnet18_lr0.5_wd5e-05_s2",
    "resnet18_lr0.5_wd0.0002_s2",
    "resnet18_lr0.5_wd0.0001_s1",
    "resnet18_lr0.125_wd0.0002_s2",
]

# (kind, param)
CONDITIONS = (
    [("base", None)]
    + [("recolor", g) for g in (2.0, 1.5, 0.75, 0.5, 0.25, 0.0)]
    + [("tot_recolor", g) for g in (0.5, 0.0)]
)


def _spec_decomp(cov, eps_rel=1e-6):
    """对称谱分解 + eps 地板（防白化爆炸）。返回 (lam_sorted_desc, V)。"""
    lam, V = np.linalg.eigh(cov)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    floor = max(float(lam[0]) * eps_rel, 1e-12)
    lam = np.clip(lam, floor, None)
    return lam, V


def fit_operators(X, y, num_classes=10):
    """ID-only：类内协方差谱（主算子）+ 总协方差谱（安慰剂）。"""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    Sw = np.zeros((X.shape[1], X.shape[1]))
    for c in range(num_classes):
        Xc = X[y == c]
        Xcc = Xc - Xc.mean(0)
        Sw += Xcc.T @ Xcc
    Sw /= max(len(X), 1)
    lam_w, V_w = _spec_decomp(Sw)
    St = np.cov(X, rowvar=False) if X.shape[1] < len(X) else Sw
    lam_t, V_t = _spec_decomp(np.asarray(St))
    return {"mu": mu, "lam_w": lam_w, "V_w": V_w,
            "lam_t": lam_t, "V_t": V_t}


def transform(X, op, kind, param):
    X = np.asarray(X, dtype=np.float64)
    if kind == "base":
        return X
    if kind == "recolor":
        lam, V = op["lam_w"], op["V_w"]
    elif kind == "tot_recolor":
        lam, V = op["lam_t"], op["V_t"]
    else:
        raise ValueError(kind)
    gain = lam ** ((param - 1.0) / 2.0)          # γ=1 → 全 1
    dX = (X - op["mu"]) @ V
    return op["mu"] + (dX * gain) @ V.T


@torch.no_grad()
def head_acc(fc, X, y, device):
    xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    return float((fc(xt).argmax(1).cpu().numpy() == np.asarray(y)).mean())


def probe_acc(Xtr, ytr, Xte, yte, num_classes, device, epochs=40, seed=0):
    return linear_probe(
        torch.as_tensor(np.asarray(Xtr), dtype=torch.float32, device=device),
        torch.as_tensor(np.asarray(ytr), dtype=torch.long, device=device),
        torch.as_tensor(np.asarray(Xte), dtype=torch.float32, device=device),
        torch.as_tensor(np.asarray(yte), dtype=torch.long, device=device),
        num_classes=num_classes, epochs=epochs, seed=seed)


def glue_on(Xg, yg, m_per_class, n_points, n_t, seed):
    mans = [Xg[c * m_per_class:(c + 1) * m_per_class].T for c in range(10)]
    sub = downsample_manifolds(mans, n_points=n_points, seed=seed)
    return glue_analysis(sub, n_t=n_t, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir",
                    default="/mnt/mydisk/xinguan-brainai/runs/ckpt_grid")
    ap.add_argument("--data",
                    default=os.environ.get("BRAINAI_DATA_ROOT",
                                           "/mnt/mydisk/xinguan-brainai/data"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "probe2"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--probe-epochs", type=int, default=40)
    ap.add_argument("--m-per-class", type=int, default=500)
    ap.add_argument("--n-points", type=int, default=50)
    ap.add_argument("--n-t", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / "intervene_recolor.json"

    models = [m for m in args.models.split(",") if m]
    rows = []
    if fp.exists():
        try:
            rows = json.load(open(fp))["rows"]
        except Exception:
            rows = []
    done = {r["model"] for r in rows if r["kind"] == "base"}
    print(f"models={len(models)} conditions={len(CONDITIONS)} done={sorted(done)}",
          flush=True)

    data = Path(args.data)
    loaders = {
        "id_tr": make_loader(str(data / "cifar_full" / "train"), 256),
        "id": make_loader(str(data / "cifar_full" / "test"), 256),
        "c100_tr": make_loader(str(data / "cifar100" / "train"), 256),
        "c100_te": make_loader(str(data / "cifar100" / "test"), 256),
    }
    merged = data / "cifar10c_merged"
    if merged.is_dir():
        loaders["c10c"] = make_loader(str(merged), 256)

    for mi, name in enumerate(models):
        if name in done:
            print(f"[skip] {name}", flush=True)
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

        Xtr, _, ytr10 = extract_features(model, loaders["id_tr"], device)
        Xid, _, yid = extract_features(model, loaders["id"], device)
        Xc_tr, _, yc_tr = extract_features(model, loaders["c100_tr"], device)
        Xc_te, _, yc_te = extract_features(model, loaders["c100_te"], device)
        Xk, _, yk = (extract_features(model, loaders["c10c"], device)
                     if "c10c" in loaders else (None, None, None))

        op = fit_operators(Xtr, ytr10)
        # 白化最大增益记录（尾部方向放大倍数）
        max_gain_w = float((op["lam_w"][0] / op["lam_w"][-1]) ** 0.5)
        Xsub, ysub = subsample_per_class(Xid, yid, args.m_per_class, 10,
                                         seed=args.seed)

        for kind, param in CONDITIONS:
            tc = time.time()
            Xsub_p = transform(Xsub, op, kind, param)
            Xid_p = transform(Xid, op, kind, param)
            Xtr_p = transform(Xtr, op, kind, param)
            Xc_tr_p = transform(Xc_tr, op, kind, param)
            Xc_te_p = transform(Xc_te, op, kind, param)

            g = glue_on(Xsub_p, ysub, args.m_per_class, args.n_points,
                        args.n_t, args.seed)
            rec = {
                "model": name, "kind": kind, "param": param,
                "glue_d": round(float(g["dimension"]), 4),
                "glue_r": round(float(g["radius"]), 4),
                "glue_u": round(float(g["utility"]), 4),
                "id_fc_acc": round(head_acc(fc, Xid_p, yid, device), 4),
                "id_probe_acc": round(probe_acc(
                    Xtr_p, ytr10, Xid_p, yid, 10, device,
                    epochs=args.probe_epochs, seed=args.seed), 4),
                "c100_probe": round(probe_acc(
                    Xc_tr_p, yc_tr, Xc_te_p, yc_te, 100, device,
                    epochs=args.probe_epochs, seed=args.seed), 4),
                "c10c_acc": (round(head_acc(
                    fc, transform(Xk, op, kind, param), yk, device), 4)
                    if Xk is not None else None),
                "max_gain": round(max_gain_w, 1),
                "lr": float(sd.get("lr", np.nan)),
                "secs": round(time.time() - tc, 1),
            }
            rows.append(rec)
            print(f"[{mi+1}/{len(models)}] {name} {kind}={param}: "
                  f"D={rec['glue_d']:.3f} idp={rec['id_probe_acc']:.4f} "
                  f"c100={rec['c100_probe']:.4f} ({rec['secs']}s)", flush=True)
            fp.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        print(f"[{mi+1}/{len(models)}] {name} done {time.time()-t0:.0f}s",
              flush=True)

    # 汇总
    from scipy.stats import pearsonr
    out = {"per_model": []}
    for m in sorted({r["model"] for r in rows}):
        pts = [r for r in rows if r["model"] == m]
        base = next((r for r in pts if r["kind"] == "base"), None)
        if base is None:
            continue
        entry = {"model": m}
        for fam in ("recolor", "tot_recolor"):
            sel = sorted([r for r in pts if r["kind"] == fam] + [base],
                         key=lambda r: -(r["param"] if r["param"] is not None
                                         else 9))
            if len(sel) < 3:
                continue
            D = np.array([r["glue_d"] for r in sel])
            Y = np.array([r["c100_probe"] for r in sel])
            Yp = np.array([r["id_probe_acc"] for r in sel])
            entry[f"{fam}_slope_c100"] = round(float(np.polyfit(D, Y, 1)[0]), 5)
            entry[f"{fam}_slope_idprobe"] = round(float(np.polyfit(D, Yp, 1)[0]), 5)
            entry[f"{fam}_manip_r"] = round(float(pearsonr(
                np.arange(len(D)) * -1.0, D)[0]), 3)
        out["per_model"].append(entry)
    (outdir / "intervene_recolor_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
