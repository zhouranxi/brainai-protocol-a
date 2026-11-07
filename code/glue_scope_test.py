#!/usr/bin/env python3
"""Gaussianization 作用域假设检验：pooled 全局 vs 按类内部。

已知：
  A(全局,无G):  dim=1.6781 rad=0.4745 util=0.1555（dim +29%, rad −9%, util −18%）
  D(按类+G):    dim=16.77 rad=0.9614 util=0.1905（util 命中，dim 崩）
  ⇒ Gaussianization 存在但作用域不是"每类内部"

假设 H：pooled whitening（线性版）——X' = X·V·Λ^{-1/2}，V/Λ 来自 pooled 协方差
假设 I：pooled Gaussianization——5000 点每维度一起做秩→高斯变换，再切回类

ref: dim=1.3009 rad=0.5240 util=0.1900
"""
import sys
import numpy as np

sys.path.insert(0, '/mnt/mydisk/xinguan-brainai/code')
from glue_impl import glue_analysis, downsample_manifolds

REF = (1.3009, 0.5240, 0.1900)


def pooled_whiten(X):
    """X: (n, d) → 白化（pooled 协方差 → 单位阵）"""
    Xc = X - X.mean(axis=0, keepdims=True)
    C = Xc.T @ Xc / (len(Xc) - 1)
    w, V = np.linalg.eigh(C)
    keep = w > 1e-10 * w.max()
    W = V[:, keep] / np.sqrt(w[keep])
    return Xc @ W


def pooled_gaussianize(X, eps=1e-12):
    """X: (n, d) 每维度 pooled 秩 → 标准高斯（Blom 映射）"""
    from scipy.stats import norm
    n, d = X.shape
    out = np.empty_like(X, dtype=np.float64)
    for j in range(d):
        v = X[:, j]
        order = np.argsort(v, kind='stable')
        ranks = np.empty(n)
        sv = v[order]
        i0 = 0
        while i0 < n:
            k = i0
            while k + 1 < n and sv[k + 1] - sv[k] <= eps:
                k += 1
            ranks[order[i0:k + 1]] = (i0 + k) / 2.0 + 1.0
            i0 = k + 1
        u = (ranks - 0.375) / (n + 0.25)
        out[:, j] = norm.ppf(u)
    return out


data = np.load('/tmp/example_feat.npy', allow_pickle=True).item()
X_all, y = data['X'], data['y']
order = np.argsort(y, kind='stable')
X_cls = X_all[order]
rng = np.random.default_rng(42)
sub_idx = np.concatenate([rng.choice(1000, 500, replace=False) + c * 1000
                          for c in range(10)])
X_sub = X_cls[sub_idx]                       # (5000, 512) class-ordered

print(f'ref: dim=1.3009 rad=0.5240 util=0.1900', flush=True)

# H：pooled whitening（500 点层做，再 downsample）
Xw = pooled_whiten(X_sub)
man_w = [Xw[c * 500:(c + 1) * 500].T for c in range(10)]
subH = downsample_manifolds(man_w, n_points=50, seed=0)
rH = glue_analysis(subH, n_t=200, seed=0)
print(f'H(pooled白化): dim={rH["dimension"]:.4f} rad={rH["radius"]:.4f} util={rH["utility"]:.4f}', flush=True)

# I：pooled Gaussianization（5000 点每维度，再切类 downsample）
Xg = pooled_gaussianize(X_sub)
man_g = [Xg[c * 500:(c + 1) * 500].T for c in range(10)]
subI = downsample_manifolds(man_g, n_points=50, seed=0)
rI = glue_analysis(subI, n_t=200, seed=0)
print(f'I(pooled高斯): dim={rI["dimension"]:.4f} rad={rI["radius"]:.4f} util={rI["utility"]:.4f}', flush=True)

# J：pooled Gaussianization 在 10000 全量做，再选 500
Xg2 = pooled_gaussianize(X_cls)
rng2 = np.random.default_rng(42)
sub_idx2 = np.concatenate([rng2.choice(1000, 500, replace=False) + c * 1000
                           for c in range(10)])
X_sub2 = Xg2[sub_idx2]
man_j = [X_sub2[c * 500:(c + 1) * 500].T for c in range(10)]
subJ = downsample_manifolds(man_j, n_points=50, seed=0)
rJ = glue_analysis(subJ, n_t=200, seed=0)
print(f'J(全量高斯):  dim={rJ["dimension"]:.4f} rad={rJ["radius"]:.4f} util={rJ["utility"]:.4f}', flush=True)

print('DONE', flush=True)
