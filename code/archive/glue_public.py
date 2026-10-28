#!/usr/bin/env python3
"""GLUE 三指标的公开实现复现（不依赖私有 glue.contrib 包）。

依据：
  - 官方仓库 analysis/geo.py 的调用协议：geo_analysis(X_sub, num_classes, M)
    → manifolds = [X[i*M:(i+1)*M].T] → downsample_manifolds(n_points=min(50,M))
    → glue_analysis(sub_manifolds) → {dimension, radius, utility}
  - GLUE 论文 bioRxiv 2024.02.26.582157 附录 B.3 的精确定义：
      D_eff := (1/P) E[b(y,t)]
      R_eff := sqrt(E[c] / E[b - c])
      Ψ_eff := E[c] / E[a]
  - Chung, Lee & Sompolinsky (PRX 2018) 的 anchor point QP 公开实现
    （chung-neuroai-lab/cshl2022-deep-learning-manifolds-tutorial）：
      minimize_vt_sq 的 cvxopt 求解 + 对偶权重 output['z'] 即 anchor 组合系数。

验证基准：官方 example_checkpoint.pth 的参考值
  dimension=1.3009  radius=0.5240  utility=0.1900
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/mydisk/xinguan-brainai/vendor/cshl2022-deep-learning-manifolds-tutorial")
from manifold_geometry.manifold_analysis import minimize_vt_sq, maxproj  # noqa: E402


def downsample_manifolds(manifolds, n_points=50, seed=0):
    """按官方 glue.preprocess.downsample_manifolds 的语义：每类等量降采样。

    manifolds: list of (n_features, n_points)
    """
    rng = np.random.default_rng(seed)
    out = []
    for m in manifolds:
        n = m.shape[1]
        if n <= n_points:
            out.append(m)
            continue
        idx = rng.choice(n, n_points, replace=False)
        out.append(m[:, idx])
    return out


def _anchor_solutions(manifolds, n_t=200, seed=0, kappa=0.0):
    """对每个 (y,t) 求解 anchor point，返回每个 t 对应的 anchor s_mu。

    对每个类 mu，用该类的样本与「其他类的负样本」构成 one-vs-rest 约束：
        y_mu · v^T x_mu(i) >= 0   (本类)
        y_mu · v^T x_nu(i) <= 0   (他类，取 -1)
    这与 GLUE 论文的 anchor 定义一致：anchor 是各类流形的凸包在
    「随机方向 t 上的支撑点」的加权组合。

    返回:
      S: (P, n_t, D1)  每类每个 t 的 anchor 向量（已含 intercept 维）
      lam: (P, n_t)    每个 t 的 anchor 组合权重（用于 R/D 的 a,b,c 计算）
    """
    rng = np.random.default_rng(seed)
    P = len(manifolds)
    D = manifolds[0].shape[0]        # 特征维
    D1 = D + 1                        # 齐次坐标维（含 intercept）
    S = np.zeros((P, n_t, D1))
    lam = np.zeros((P, n_t))

    # 采样 n_t 个随机方向（D1 维，与 sD1 对齐）
    t_vecs = rng.standard_normal((D1, n_t))

    for mu in range(P):
        m_mu = manifolds[mu]
        D = D1 - 1
        # 拼接本类点（label +1）与他类点（label -1）
        pos = m_mu.T                    # (n_mu, D)
        neg = np.concatenate([manifolds[k].T for k in range(P) if k != mu], axis=0)
        y = np.concatenate([np.ones(len(pos)), -np.ones(len(neg))])
        S_all = np.concatenate([pos, neg], axis=0)      # (N, D)
        sD1 = np.vstack([S_all.T, np.ones((1, len(S_all)))])   # (D+1, N)

        for i in range(n_t):
            t = t_vecs[:, i:i + 1]
            # maxproj 给出该方向上支撑点，若其标签为 -1 则 v=t 无约束解
            ss, gg = maxproj(t_vecs[:, i:i + 1], sD1)
            if gg[0] < 0:
                # 该方向上本类不占优 → anchor 在他类方向，无有效分离
                S[mu, i, :] = ss[:, 0]
                lam[mu, i] = 0.0
                continue
            v_f, _, exitflag, alpha, normvt2 = minimize_vt_sq(t, sD1, kappa=kappa)
            if exitflag != 1:
                S[mu, i, :] = ss[:, 0]
                lam[mu, i] = 0.0
                continue
            scale = np.sum(alpha)
            if abs(scale) < 1e-12:
                S[mu, i, :] = ss[:, 0]
                lam[mu, i] = 0.0
            else:
                s_f = (t - v_f) / scale
                S[mu, i, :] = s_f[:, 0]
                lam[mu, i] = scale
    return S, lam, t_vecs


def glue_analysis(manifolds, n_t=200, seed=0, kappa=0.0):
    """GLUE 三指标（dimension / radius / utility）。

    按论文附录 B.3：
      a(y,t) = ||s_mu(y,t) - s_mu_0(y,t)||^2    （轴向分量的模长）
      b(y,t) = <s_mu(y,t), s_mu(y,t)>            （anchor 点的模长平方）
      c(y,t) = <t, s_mu(y,t)>                   （响应）
      D_eff = (1/P) E[b]
      R_eff = sqrt(E[c] / E[b - c])
      Ψ_eff = E[c] / E[a]
    其中 s_mu_0 = E_t[s_mu(t)]（anchor center）。
    """
    P = len(manifolds)
    S, _, t_vecs = _anchor_solutions(manifolds, n_t=n_t, seed=seed, kappa=kappa)

    # s_mu_0：每类 anchor 的均值
    S0 = S.mean(axis=1)                                  # (P, D1)
    # 轴向分量
    Saxis = S - S0[:, None, :]                           # (P, n_t, D1)

    # a：轴向分量的模长平方（去掉 center 维）
    a = np.sum(Saxis[:, :, :-1] ** 2, axis=2)            # (P, n_t)
    # b：anchor 模长平方
    b = np.sum(S[:, :, :-1] ** 2, axis=2)                # (P, n_t)
    # c：响应 = <t, s>（复用求解时的同一批随机方向）
    # t_vecs: (D+1, n_t) → 取前 D 维并转置为 (1, D, n_t) 以广播
    tv = t_vecs[:-1, :].T[None, :, :]                      # (1, D, n_t)
    c = np.sum(S[:, :, :-1] * tv, axis=2)                   # (P, n_t)

    dimension = float(np.mean(b))
    denom = float(np.mean(b - c))
    radius = float(np.sqrt(np.mean(c) / denom)) if denom > 0 else float("nan")
    denom_a = float(np.mean(a))
    utility = float(np.mean(c) / denom_a) if denom_a > 0 else float("nan")
    return {"dimension": dimension, "radius": radius, "utility": utility}


def geo_analysis(X, num_classes, M, seed=0, n_t=200, rcond=1e-3):
    """与官方 analysis/geo.py 同签名。"""
    from sklearn.decomposition import PCA

    manifolds = [X[i * M:(i + 1) * M].T for i in range(num_classes)]
    sub = downsample_manifolds(manifolds, n_points=min(50, M), seed=seed)
    glue_ret = glue_analysis(sub, n_t=n_t, seed=seed)

    pca = PCA().fit(X)
    ev = pca.explained_variance_
    pr = float((ev.sum() ** 2) / np.sum(ev ** 2))
    return {
        "participation_ratio": pr,
        "dimension": glue_ret["dimension"],
        "radius": glue_ret["radius"],
        "utility": glue_ret["utility"],
    }


if __name__ == "__main__":
    import argparse
    import json

    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/mydisk/xinguan-brainai/vendor/ood-generalization-geometry/data/example_checkpoint.pth")
    ap.add_argument("--data", default="/mnt/mydisk/xinguan-brainai/data/cifar_full/test")
    ap.add_argument("--n-sub", type=int, default=500)
    ap.add_argument("--n-t", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="/mnt/mydisk/xinguan-brainai/results/glue_public_validation.json")
    args = ap.parse_args()

    VENDOR = "/mnt/mydisk/xinguan-brainai/vendor/ood-generalization-geometry"
    sys.path.insert(0, VENDOR)
    from models import build_model

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu")
    m = build_model("ResNet18", dataset="cifar").to(dev)
    m.load_state_dict(ck["model"] if "model" in ck else ck)
    m.eval()

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    ds = datasets.ImageFolder(args.data, transform=tf)

    buf, ys = [], []
    h = m.avgpool.register_forward_hook(
        lambda mo, i, o: buf.append(torch.flatten(o, 1).detach().cpu()))
    with torch.no_grad():
        for xb, yb in DataLoader(ds, batch_size=256, shuffle=False, num_workers=4):
            m(xb.to(dev))
            ys.append(yb.numpy())
    h.remove()
    X = torch.cat(buf).numpy()
    y = np.concatenate(ys)
    o = np.argsort(y, kind="stable")
    X, y = X[o], y[o]

    rng = np.random.default_rng(42)
    idx = np.concatenate([rng.choice(1000, args.n_sub, replace=False) + c * 1000
                           for c in range(10)])
    Xs = X[idx]

    REF = {"dimension": 1.3009, "radius": 0.5240, "utility": 0.1900}
    print(f"{'seed':>5} {'dimension':>11} {'radius':>9} {'utility':>9} {'PR':>9}")
    print(f"{'REF':>5} {REF['dimension']:>11.4f} {REF['radius']:>9.4f} "
          f"{REF['utility']:>9.4f} {'-':>9}")

    recs = []
    for sd in range(args.seeds):
        r = geo_analysis(Xs, 10, args.n_sub, seed=sd, n_t=args.n_t)
        recs.append(r)
        print(f"{sd:>5} {r['dimension']:>11.4f} {r['radius']:>9.4f} "
              f"{r['utility']:>9.4f} {r['participation_ratio']:>9.4f}")

    arr = {k: np.array([r[k] for r in recs]) for k in recs[0]}
    print()
    print("=== 与官方参考值对比 ===")
    for k in ["dimension", "radius", "utility"]:
        v = arr[k]
        rel = (v.mean() - REF[k]) / REF[k] * 100
        print(f"  {k:10s} 官方={REF[k]:.4f}  本实现={v.mean():.4f}±{v.std():.4f}  "
              f"相对偏差={rel:+.1f}%")

    with open(args.out, "w") as f:
        json.dump({"records": recs, "reference": REF}, f, indent=2)
    print(f"\n[saved] {args.out}")
