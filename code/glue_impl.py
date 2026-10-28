#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GLUE 三指标（D_eff / R_eff / Ψ_eff）的忠实实现。

依据：
  - Chou et al. 2025 (bioRxiv 2024.02.26.582157v2) Methods M3.3 + Algorithm 1
  - Chou et al. 2026 (ICLR, arXiv 2603.01879) Appendix B.3.1
  - 官方仓库 chung-neuroai-lab/ood-generalization-geometry 的 analysis/geo.py

⚠️ 官方 GLUE 包（`glue.contrib`）需向作者申请 early access，本文件是
   **依据论文伪代码的独立实现**。正确性用官方仓库提供的
   `data/example_checkpoint.pth` 的参考数值验证：
       participation_ratio = 10.2586, neural_collapse = 0.2104,
       dimension = 1.3009, radius = 0.5240, utility = 0.1900
   （官方 example.ipynb 输出，n_sub=500/class，10 类）

Algorithm 1：
  Step 1  采样 nt=200 个 (y, t)；解 QP 得对偶支持向量权重 z_dual，
          anchor point s_μ[k] = Σ_i z_{μ,i} · x_{μ,i} / Σ_i z_{μ,i}
  Step 2  s_μ0 = mean_k s_μ[k]
  Step 3  s_μ1[k] = s_μ[k] − s_μ0
  Step 4  a = (St)ᵀ(SSᵀ)†(St)；b = (S1t)ᵀ(S1S1ᵀ)†(S1t)；
          c = (S1t)ᵀ(S0S0ᵀ + S1S1ᵀ)†(S1t)
          D = (1/P) E[b]；R = sqrt(E[c]/E[b−c])；Ψ = E[c]/E[a]；α = (1/P) E[a]
"""
from __future__ import annotations

import numpy as np

try:
    from cvxopt import matrix, solvers
    _HAS_CVXOPT = True
except Exception:                                       # pragma: no cover
    _HAS_CVXOPT = False

if _HAS_CVXOPT:
    solvers.options["show_progress"] = False
    solvers.options["maxiters"] = 1000000
    solvers.options["abstol"] = 1e-10
    solvers.options["reltol"] = 1e-10
    solvers.options["feastol"] = 1e-10


# --------------------------------------------------------------------------- #
def downsample_manifolds(manifolds, n_points, seed=0):
    """等价于官方 `glue.preprocess.downsample_manifolds`：每条流形随机取 n_points 点。"""
    rng = np.random.default_rng(seed)
    out = []
    for m in manifolds:
        n_pts = m.shape[1]
        if n_pts <= n_points:
            out.append(m)
        else:
            idx = rng.choice(n_pts, n_points, replace=False)
            out.append(m[:, idx])
    return out


def _pinv_psd(M, rcond=1e-10):
    """对称半正定矩阵的伪逆（特征值截断）。"""
    w, V = np.linalg.eigh((M + M.T) / 2.0)
    wmax = w.max()
    if wmax <= 0:
        return np.zeros_like(M)
    keep = w > rcond * wmax
    if not np.any(keep):
        return np.zeros_like(M)
    return (V[:, keep] / w[keep]) @ V[:, keep].T


def _one_vs_rest_dichotomies(P):
    """one-vs-rest 二分集合：第 i 个为 +1，其余 −1。"""
    Y = np.full((P, P), -1.0)
    np.fill_diagonal(Y, 1.0)
    return Y


def glue_analysis(manifolds, n_t=200, kappa=0.0, seed=0,
                  dichotomies="one_vs_rest", n_pairs=None):
    """按 Algorithm 1 计算 GLUE 指标。

    manifolds : list of (N, M_mu) 数组（论文约定：特征维 × 点数）
    n_t       : 采样的 (y, t) 对数（论文用 200）
    kappa     : margin（论文用 0）
    dichotomies : "one_vs_rest" | "random"

    返回 dict：alpha, dimension(D_eff), radius(R_eff), utility(Ψ_eff)
    """
    if not _HAS_CVXOPT:
        raise RuntimeError("cvxopt required for GLUE analysis")

    P = len(manifolds)
    N = manifolds[0].shape[0]
    rng = np.random.default_rng(seed)
    Y = _one_vs_rest_dichotomies(P) if dichotomies == "one_vs_rest" else None

    M_list = [m.shape[1] for m in manifolds]
    offsets = np.cumsum([0] + M_list)

    A = np.eye(N)

    S = np.zeros((n_t, P, N))
    Ts = np.zeros((n_t, N))

    for k in range(n_t):
        t = rng.standard_normal(N)
        Ts[k] = t
        if Y is not None:
            y = Y[k % P]
        else:
            y = rng.choice([-1.0, 1.0], size=P)

        # G：(sum M, N)，第 (μM+i) 行 = y_μ · x_{μ,i}
        G = np.concatenate([y[mu] * manifolds[mu].T for mu in range(P)], axis=0)
        h = np.full(G.shape[0], kappa)
        q = -t

        try:
            sol = solvers.qp(matrix(A), matrix(q), matrix(G), matrix(h))
            # cvxopt 的对偶变量在 sol['z']（sol['y'] 是等式约束的拉格朗日乘子，
            # 无等式约束时是空数组）。Algorithm 1 里的 z_dual 即此处的 z。
            z = np.array(sol["z"]).flatten()
        except Exception:
            z = np.zeros(G.shape[0])

        for mu in range(P):
            zmu = z[offsets[mu]:offsets[mu + 1]]
            den = zmu.sum()
            if abs(den) < 1e-12:
                S[k, mu] = manifolds[mu].mean(axis=1)
            else:
                S[k, mu] = (zmu @ manifolds[mu].T) / den

    # Step 2/3
    S0 = S.mean(axis=0)                     # (P, N)
    S1 = S - S0[None, :, :]                 # (nt, P, N)

    a_l, b_l, c_l = [], [], []
    for k in range(n_t):
        Sk, S1k, t = S[k], S1[k], Ts[k]

        St = Sk @ t                          # (P,)
        S1t = S1k @ t

        a = float(St @ _pinv_psd(Sk @ Sk.T) @ St)
        b = float(S1t @ _pinv_psd(S1k @ S1k.T) @ S1t)
        c = float(S1t @ _pinv_psd(S0 @ S0.T + S1k @ S1k.T) @ S1t)

        a_l.append(a); b_l.append(b); c_l.append(c)

    a_m = float(np.mean(a_l))
    b_m = float(np.mean(b_l))
    c_m = float(np.mean(c_l))
    denom = b_m - c_m

    out = {
        "alpha": (1.0 / P) / a_m if a_m > 0 else float("nan"),
        "dimension": (1.0 / P) * b_m,
        "radius": float(np.sqrt(c_m / denom)) if denom > 0 else float("nan"),
        "utility": c_m / a_m if a_m > 0 else float("nan"),
    }
    return out


# --------------------------------------------------------------------------- #
# 与官方 geo.py 一致的两个纯 numpy 指标
# --------------------------------------------------------------------------- #
def compute_participation_ratio_global(features):
    """官方 analysis/geo.py 的 compute_participation_ratio（全局 pooled）。"""
    x = features - features.mean(axis=0, keepdims=True)
    C = (x.T @ x) / max(len(x) - 1, 1)
    lam = np.linalg.eigvalsh(C)
    lam = lam[lam > 0]
    if lam.size == 0:
        return float("nan")
    return float((lam.sum() ** 2) / np.sum(lam ** 2))


def compute_participation_ratio_perclass(features, num_classes, M):
    """论文附录 B.3 式 (3)：逐类 PR 后取平均。"""
    vals = []
    for c in range(num_classes):
        Xc = features[c * M:(c + 1) * M]
        if len(Xc) < 2:
            continue
        Xc = Xc - Xc.mean(axis=0, keepdims=True)
        C = (Xc.T @ Xc) / (len(Xc) - 1)
        lam = np.linalg.eigvalsh(C)
        lam = lam[lam > 0]
        if lam.size:
            vals.append((lam.sum() ** 2) / np.sum(lam ** 2))
    return float(np.mean(vals)) if vals else float("nan")


def compute_neural_collapse(features, num_classes, M, rcond=1e-3):
    """官方 analysis/geo.py 的 neural_collapse_1（逐字移植）。

    ⚠️ 官方签名是 `neural_collapse_1(features, num_classes, M)`，其中 features
    形状为 **(N_features, N_samples)**（官方 geo_analysis 内部传的是 `X.T`）。
    本实现保持官方约定：调用方需传 `features.T`。
    """
    X_dict = {c: features[:, c * M:(c + 1) * M] for c in range(num_classes)}
    X_centered = {c: X_dict[c] - X_dict[c].mean(axis=1, keepdims=True)
                  for c in range(num_classes)}
    Sigma_dict = {c: X_centered[c] @ X_centered[c].T for c in range(num_classes)}
    Sigma_W = sum(Sigma_dict.values()) / (num_classes * M)
    mu_dict = {c: np.mean(X_dict[c], axis=1) for c in range(num_classes)}
    mu_G = np.mean(features, axis=1)          # features: (N_feat, N_samples)
    Sigma_B = sum(np.outer(mu_dict[c] - mu_G, mu_dict[c] - mu_G)
                  for c in range(num_classes)) / num_classes
    eigvals, eigvecs = np.linalg.eigh(Sigma_B)
    lam_max = eigvals.max()
    if lam_max <= 0:
        return 0.0
    keep = eigvals > rcond * lam_max
    if not np.any(keep):
        return 0.0
    et, ut = eigvals[keep], eigvecs[:, keep]
    pinv = (ut / et) @ ut.T
    pinv = 0.5 * (pinv + pinv.T)
    return float(np.trace(Sigma_W @ pinv) / num_classes)


if __name__ == "__main__":
    # 合成自检：P 个正交球，应给出合理的 D/R/Ψ
    rng = np.random.default_rng(0)
    P, D, M = 4, 20, 50
    manifolds = []
    for p in range(P):
        c = np.zeros(D); c[p * 4] = 5.0
        pts = c[:, None] + rng.standard_normal((D, M)) * 0.5
        manifolds.append(pts)
    res = glue_analysis(manifolds, n_t=50, seed=0)
    print("synthetic:", {k: round(v, 4) for k, v in res.items()})
    print("expect: dimension 远小于 D, utility in (0,1)")
