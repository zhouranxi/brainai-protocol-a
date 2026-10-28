#!/usr/bin/env python3
"""GLUE 公开实现的规格逆向：扫描关键设计选择，对标官方参考值。

官方 example_checkpoint.pth 参考值：dimension=1.3009 radius=0.5240 utility=0.1900

扫描维度：
  A. 约束构造方式：
     A1 全 P 流形同时进约束（y_mu 符号法，Chung 2018 原版）
     A2 one-vs-rest（每类单独 vs 其余）
  B. anchor 是否含 intercept 维（D+1 齐次坐标 vs 仅 D 维）
  C. c 的定义：<t, s>（含 intercept 维 vs 仅特征维）
  D. 降采样点数（50 = 官方默认）
  E. n_t（200 = 官方默认）
  F. kappa（margin）
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

VENDOR = "/mnt/mydisk/xinguan-brainai/vendor/cshl2022-deep-learning-manifolds-tutorial"
sys.path.insert(0, VENDOR)
sys.path.insert(0, "/mnt/mydisk/xinguan-brainai/vendor/ood-generalization-geometry")
from manifold_geometry.manifold_analysis import minimize_vt_sq, maxproj  # noqa: E402


def downsample(manifolds, n_points, seed):
    rng = np.random.default_rng(seed)
    out = []
    for m in manifolds:
        if m.shape[1] <= n_points:
            out.append(m)
        else:
            idx = rng.choice(m.shape[1], n_points, replace=False)
            out.append(m[:, idx])
    return out


def build_constraints(manifolds, mode, intercept):
    """返回 (sD1, y) —— sD1: (D1, N)，y: (N,) 标签"""
    P = len(manifolds)
    D = manifolds[0].shape[0]
    if mode == "all":
        pts, ys = [], []
        for mu in range(P):
            pts.append(manifolds[mu].T)
            ys.append(np.full(manifolds[mu].shape[1], 1.0 if mu == 0 else -1.0))
        S_all = np.concatenate(pts, axis=0)
        y = np.concatenate(ys)
    elif mode == "ovr":
        pts = [manifolds[0].T]
        ys = [np.ones(manifolds[0].shape[1])]
        for mu in range(1, P):
            pts.append(manifolds[mu].T)
            ys.append(-np.ones(manifolds[mu].shape[1]))
        S_all = np.concatenate(pts, axis=0)
        y = np.concatenate(ys)
    if intercept:
        sD1 = np.vstack([S_all.T, np.ones((1, len(S_all)))])
    else:
        sD1 = S_all.T
    return sD1, y


def solve_all(manifolds, mode, intercept, n_t, seed, kappa):
    """Chung 2018 风格：对每个 t 求解 v 与 anchor 组合。

    约束： y_i · (v·x_i) >= kappa
    anchor: s = Σ z_i x_i / Σ z_i（z 为对偶权重）
    """
    P = len(manifolds)
    D = manifolds[0].shape[0]
    D1 = D + 1
    sD1, y = build_constraints(manifolds, mode, intercept)
    N = sD1.shape[1]

    rng = np.random.default_rng(seed)
    t_vecs = rng.standard_normal((D1, n_t))

    S = np.zeros((P, n_t, D1))
    for i in range(n_t):
        t = t_vecs[:, i:i + 1]
        # 求 v：min ||v-t||^2 s.t.  y·v·x >= kappa
        # → G v >= h,  G = diag(y) @ sD1.T ... 实际用 v 的前 D 维与 x 点积
        G = (y[:, None] * sD1[:-1, :].T)      # (N, D)  若 intercept 则去掉最后一行
        h = np.full(N, kappa)
        from cvxopt import matrix, solvers
        solvers.options['show_progress'] = False
        try:
            out = solvers.qp(matrix(np.identity(D)), matrix(-t[:D].astype(float)),
                             matrix(G.astype(float)), matrix(h.astype(float)))
        except Exception:
            continue
        if out['status'] != 'optimal':
            continue
        v = np.array(out['x']).flatten()
        z = np.array(out['z']).flatten()
        scale = z.sum()
        if abs(scale) < 1e-12:
            continue
        # anchor = (t - v)/scale  （与 GLUE 论文 Algorithm 1 一致）
        s = (t[:, 0] - np.concatenate([v, [0.0]])) / scale
        # 记录到"本类"上：one-vs-rest 模式下第 0 类是目标类
        S[0, i, :] = s
        # 其余类用同一 v 但取其对应支撑方向近似：直接置为该类点的最大投影
        for mu in range(1, P):
            mm = manifolds[mu]
            proj = mm.T @ v
            S[mu, i, :-1] = mm.T[np.argmax(proj)]
            S[mu, i, -1] = 1.0
    return S, t_vecs


def metrics(S, t_vecs, mode):
    P, n_t, D1 = S.shape
    D = D1 - 1
    S0 = S.mean(axis=1)
    Saxis = S - S0[:, None, :]
    a = np.sum(Saxis[:, :, :D] ** 2, axis=2)
    b = np.sum(S[:, :, :D] ** 2, axis=2)
    tv = t_vecs[:D, :].T[None, :, :]
    c = np.sum(S[:, :, :D] * tv, axis=2)

    dim = float(np.mean(b))
    denom = float(np.mean(b - c))
    rad = float(np.sqrt(np.mean(c) / denom)) if denom > 0 else float("nan")
    da = float(np.mean(a))
    ut = float(np.mean(c) / da) if da > 0 else float("nan")
    return {"dimension": dim, "radius": rad, "utility": ut}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from models import build_model

    V = "/mnt/mydisk/xinguan-brainai/vendor/ood-generalization-geometry"
    sys.path.insert(0, V)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(f"{V}/data/example_checkpoint.pth", map_location="cpu")
    m = build_model("ResNet18", dataset="cifar").to(dev)
    m.load_state_dict(ck["model"] if "model" in ck else ck)
    m.eval()
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                  (0.2023, 0.1994, 0.2010))])
    ds = datasets.ImageFolder("/mnt/mydisk/xinguan-brainai/data/cifar_full/test", transform=tf)
    buf, ys = [], []
    h = m.avgpool.register_forward_hook(lambda mo, i, o: buf.append(torch.flatten(o, 1).detach().cpu()))
    with torch.no_grad():
        for xb, yb in DataLoader(ds, batch_size=256, shuffle=False, num_workers=4):
            m(xb.to(dev)); ys.append(yb.numpy())
    h.remove()
    X = torch.cat(buf).numpy(); y = np.concatenate(ys)
    o = np.argsort(y, kind="stable"); X, y = X[o], y[o]
    rng = np.random.default_rng(42)
    idx = np.concatenate([rng.choice(1000, 500, replace=False) + c * 1000 for c in range(10)])
    Xs = X[idx]

    REF = {"dimension": 1.3009, "radius": 0.5240, "utility": 0.1900}
    print(f"官方参考: dim={REF['dimension']} rad={REF['radius']} util={REF['utility']}\n")
    print(f"{'mode':>6} {'iccpt':>6} {'npts':>5} {'n_t':>5} {'kappa':>6} "
          f"{'dim':>9} {'rad':>8} {'util':>8}  {'d_err%':>7} {'r_err%':>7} {'u_err%':>7}")
    print("-" * 95)

    best = []
    for mode, iccpt, npts, n_t, kappa in itertools.product(
            ["all", "ovr"], [True, False], [50], [200], [0.0]):
        dims, rads, utils = [], [], []
        for sd in range(args.seeds):
            mans = [Xs[c * 500:(c + 1) * 500].T for c in range(10)]
            sub = downsample(mans, npts, sd)
            try:
                S, tv = solve_all(sub, mode, iccpt, n_t, sd, kappa)
                mm = metrics(S, tv, mode)
            except Exception as e:
                dims = rads = utils = [float("nan")]
                break
            dims.append(mm["dimension"]); rads.append(mm["radius"]); utils.append(mm["utility"])
        d, r, u = np.nanmean(dims), np.nanmean(rads), np.nanmean(utils)
        de = (d - REF["dimension"]) / REF["dimension"] * 100
        re = (r - REF["radius"]) / REF["radius"] * 100
        ue = (u - REF["utility"]) / REF["utility"] * 100
        print(f"{mode:>6} {str(iccpt):>6} {npts:>5} {n_t:>5} {kappa:>6.2f} "
              f"{d:>9.4f} {r:>8.4f} {u:>8.4f}  {de:>+7.1f} {re:>+7.1f} {ue:>+7.1f}")
        best.append((abs(de) + abs(re) + abs(ue), mode, iccpt, d, r, u))

    best.sort()
    print("\n=== 按总相对误差排序 top3 ===")
    for b in best[:3]:
        print(f"  总误差={b[0]:.1f}%  mode={b[1]} intercept={b[2]}  "
              f"dim={b[3]:.4f} rad={b[4]:.4f} util={b[5]:.4f}")


if __name__ == "__main__":
    main()
