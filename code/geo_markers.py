#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公开可复现的几何标记实现（R1）

对应 ICLR 2026《Diagnosing Generalization Failures from Representational Geometry Markers》
附录 B.3 的定义，以及官方仓库 analysis/geo.py 中不依赖私有 GLUE 包的部分。

**重要差异说明**
官方 `analysis/geo.py` 的 `compute_participation_ratio(X)` 是**全局** pooled 协方差，
但论文附录 B.3 明确写的是**逐类计算再平均**：
    D_PR^μ = (Σλ_i)² / Σ(λ_i²)，  PR = (1/P) Σ_μ D_PR^μ
两者不是一回事。本模块**同时实现两者**（pr_global / pr_perclass），
并按论文定义把 pr_perclass 作为主口径。

已实现的标记（全部无需私有包）
- pr_perclass  : participation ratio，逐类算后平均（论文主口径，task-agnostic）
- pr_global    : participation ratio，全局 pooled（官方 geo.py 口径）
- nc1          : Neural Collapse 零塌缩 (1/P) tr(Σ_W Σ_B†)，τ=1e-3 截断伪逆
- numerical_rank: 逐类协方差奇异值的数值秩，τ=1e-3，类间平均
- local_smoothness : 流形局部光滑度（沿用 Phase 0 定义，保证可比）
- effective_dim_pca: 全局 PCA 累计 90% 方差维度（沿用 Phase 0 定义）

**未实现**：GLUE 的 D_eff / R_eff / Ψ_eff —— 需要私有 `glue.contrib` 包
（需向作者申请 early access）。本模块不伪造这三个指标。

用法：
  from geo_markers import compute_all
  m = compute_all(X, labels, num_classes, M)   # X 按类连续排列
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# participation ratio
# --------------------------------------------------------------------------- #
def pr_perclass(X: np.ndarray, num_classes: int, M: int) -> float:
    """逐类 participation ratio 后取平均（论文附录 B.3 式 (3) 的口径）。

    X: (P*M, N)，按类连续排列：类 μ 占 X[μ*M:(μ+1)*M]
    """
    vals = []
    for c in range(num_classes):
        Xc = X[c * M:(c + 1) * M]
        if Xc.shape[0] < 2:
            continue
        Xc = Xc - Xc.mean(axis=0, keepdims=True)
        # 协方差特征值
        C = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
        try:
            lam = np.linalg.eigvalsh(C)
        except np.linalg.LinAlgError:
            continue
        lam = lam[lam > 0]
        if lam.size == 0:
            continue
        vals.append((lam.sum() ** 2) / np.sum(lam ** 2))
    return float(np.mean(vals)) if vals else float("nan")


def pr_global(X: np.ndarray) -> float:
    """全局 pooled participation ratio（官方 analysis/geo.py 的口径）。

    注意：与论文附录 B.3 的逐类口径不同 —— 两者会因类间分离而显著不同。
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        lam = np.linalg.eigvalsh((Xc.T @ Xc) / max(Xc.shape[0] - 1, 1))
    except np.linalg.LinAlgError:
        return float("nan")
    lam = lam[lam > 0]
    if lam.size == 0:
        return float("nan")
    return float((lam.sum() ** 2) / np.sum(lam ** 2))


# --------------------------------------------------------------------------- #
# Neural Collapse NC1
# --------------------------------------------------------------------------- #
def nc1(X: np.ndarray, num_classes: int, M: int, rcond: float = 1e-3) -> float:
    """NC1 = (1/P) tr(Σ_W Σ_B†)，Σ_B 用截断伪逆（τ=1e-3）。

    与官方 analysis/geo.py 的 neural_collapse_1 一致（该函数是纯 numpy，无 GLUE 依赖）。
    """
    Xd = {c: X[c * M:(c + 1) * M] for c in range(num_classes)}
    Xc = {c: Xd[c] - Xd[c].mean(axis=0, keepdims=True) for c in range(num_classes)}
    Sigma_W = sum((Xc[c].T @ Xc[c]) for c in range(num_classes)) / (num_classes * M)

    mu = {c: Xd[c].mean(axis=0) for c in range(num_classes)}
    mu_G = X.mean(axis=0)
    Sigma_B = sum(np.outer(mu[c] - mu_G, mu[c] - mu_G) for c in range(num_classes)) / num_classes

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


# --------------------------------------------------------------------------- #
# numerical rank (Tunnel Effect)
# --------------------------------------------------------------------------- #
def numerical_rank(X: np.ndarray, num_classes: int, M: int, tau: float = 1e-3) -> float:
    """逐类协方差奇异值的数值秩，类间平均（论文附录 B.3）。"""
    ranks = []
    for c in range(num_classes):
        Xc = X[c * M:(c + 1) * M]
        if Xc.shape[0] < 2:
            continue
        Xc = Xc - Xc.mean(axis=0, keepdims=True)
        sv = np.linalg.svd(Xc, compute_uv=False)
        if sv.size == 0 or sv[0] <= 0:
            continue
        ranks.append(int((sv >= tau * sv[0]).sum()))
    return float(np.mean(ranks)) if ranks else float("nan")


# --------------------------------------------------------------------------- #
# Phase 0 沿用指标（保证跨实验可比）
# --------------------------------------------------------------------------- #
def effective_dim_pca(X: np.ndarray, var_ratio: float = 0.90) -> float:
    """全局 PCA 累计方差达 var_ratio 所需维度（Phase 0 同一定义，用于跨实验对照）。"""
    Xc = X - X.mean(axis=0, keepdims=True)
    n = Xc.shape[0]
    if n > 2000:
        idx = np.random.RandomState(0).choice(n, 2000, replace=False)
        Xc = Xc[idx]
    try:
        s = np.linalg.svd(Xc, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    if s.sum() <= 0:
        return float("nan")
    cum = np.cumsum(s ** 2) / np.sum(s ** 2)
    return float(np.searchsorted(cum, var_ratio) + 1)


def local_smoothness(X: np.ndarray, k: int = 10) -> float:
    """流形局部光滑度代理（Phase 0 同一定义）。"""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    n = Xn.shape[0]
    if n <= k + 1:
        return float("nan")
    sq = (Xn * Xn).sum(1)
    dists = []
    for i in range(0, n, 512):
        blk = Xn[i:i + 512]
        d2 = sq[i:i + 512, None] + sq[None, :] - 2.0 * (blk @ Xn.T)
        d2[np.arange(blk.shape[0]), np.arange(i, i + blk.shape[0])] = np.inf
        dists.append(np.sqrt(np.maximum(np.partition(d2, k, axis=1)[:, :k], 0)))
    return float(np.concatenate(dists, axis=0).mean())


def class_separability(X: np.ndarray, labels: np.ndarray) -> float:
    """类间可分性（组间方差/总方差）。使用标签 → 属平凡类，仅作对照。"""
    Xc = X - X.mean(axis=0, keepdims=True)
    total = (Xc ** 2).sum()
    if total <= 0:
        return float("nan")
    between = 0.0
    for c in np.unique(labels):
        Xk = Xc[labels == c]
        between += len(Xk) * ((Xk.mean(axis=0)) ** 2).sum()
    return float(between / total)


# --------------------------------------------------------------------------- #
def compute_all(X: np.ndarray, labels: np.ndarray, num_classes: int, M: int) -> dict:
    """一次性算出所有公开标记。

    X: (P*M, N)，**按类连续排列**（类 μ 在 [μ*M:(μ+1)*M)）
    labels: 与 X 对应的标签（用于可分性对照）
    """
    out = {
        "pr_perclass": pr_perclass(X, num_classes, M),
        "pr_global": pr_global(X),
        "nc1": nc1(X, num_classes, M),
        "numerical_rank": numerical_rank(X, num_classes, M),
        "effective_dim_pca": effective_dim_pca(X),
        "local_smoothness": local_smoothness(X),
        "class_separability": class_separability(X, labels),
    }
    return out


if __name__ == "__main__":
    # 自检：合成数据下 PR 应接近真实内在维度
    rng = np.random.RandomState(0)
    d = 5
    Z = rng.randn(1000, d)                       # 5 维内在结构
    Q = np.linalg.qr(rng.randn(d, 50))[0]        # 旋转到 50 维
    X = Z @ Q.T
    Xc = X - X.mean(0, keepdims=True)
    lam = np.linalg.eigvalsh((Xc.T @ Xc) / (len(X) - 1))
    lam = lam[lam > 0]
    pr = (lam.sum() ** 2) / np.sum(lam ** 2)
    print(f"合成 5 维流形 → PR = {pr:.2f}（期望 ≈5）")

    # 逐类版本
    Xc2 = np.repeat(np.eye(10), 100, axis=0) * 3.0   # 10 个点，每类 100 样本
    labels = np.repeat(np.arange(10), 100)
    Xc2 += rng.randn(*Xc2.shape) * 0.1
    print("逐类 PR:", round(pr_perclass(Xc2, 10, 100), 3),
          "全局 PR:", round(pr_global(Xc2), 3),
          "NC1:", round(nc1(Xc2, 10, 100), 4),
          "numerical rank:", numerical_rank(Xc2, 10, 100))
