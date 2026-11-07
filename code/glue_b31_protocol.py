#!/usr/bin/env python3
"""按论文 B.3.1 协议补齐两个缺失步骤后重验官方 example。

缺失协议（论文 B.3.1 原文）：
  "for each manifold we subsample to 50 points, conduct GLUE analysis
   on each manifold pair, and apply Gaussianization preprocessing"

1. pairwise：对 C(P,2) 对 manifold 分别跑 GLUE，再对结果取平均
2. Gaussianization：每个 manifold 的每个特征维度做单调高斯化
  （rank → 标准 normal 分位数），确保初始线性可分

对照：官方 example 参考值 dim=1.3009 rad=0.5240 util=0.1900
"""
import sys
import numpy as np
from itertools import combinations

sys.path.insert(0, '/mnt/mydisk/xinguan-brainai/code')
from glue_impl import glue_analysis, downsample_manifolds

REF = (1.3009, 0.5240, 0.1900)


def gaussianize(manifold, eps=1e-12):
    """每特征维度单调高斯化：rank → 标准 normal 分位数。

    manifold: (N, M) — 特征维 × 点数
    """
    from scipy.stats import norm
    N, M = manifold.shape
    out = np.empty_like(manifold, dtype=np.float64)
    for i in range(N):
        v = manifold[i]
        # 并列值取平均秩（ties → midrank）
        order = np.argsort(v, kind='stable')
        ranks = np.empty(M)
        sv = v[order]
        i0 = 0
        while i0 < M:
            j = i0
            while j + 1 < M and sv[j + 1] - sv[j] <= eps:
                j += 1
            mid = (i0 + j) / 2.0 + 1.0          # 1-based midrank
            ranks[order[i0:j + 1]] = mid
            i0 = j + 1
        # Blom 映射：(rank - 3/8) / (M + 1/4) → norm.ppf
        u = (ranks - 0.375) / (M + 0.25)
        out[i] = norm.ppf(u)
    return out


data = np.load('/tmp/example_feat.npy', allow_pickle=True).item()
X_all, y = data['X'], data['y']
order = np.argsort(y, kind='stable')
X_cls = X_all[order]
rng = np.random.default_rng(42)
sub_idx = np.concatenate([rng.choice(1000, 500, replace=False) + c * 1000
                          for c in range(10)])
X_sub = X_cls[sub_idx]
manifolds500 = [X_sub[c * 500:(c + 1) * 500].T for c in range(10)]

print(f'ref: dim=1.3009 rad=0.5240 util=0.1900', flush=True)

# —— 协议 A：当前实现（全局 10 manifold、无高斯化、50 点）＝ 对照
sub50 = downsample_manifolds(manifolds500, n_points=50, seed=0)
rA = glue_analysis(sub50, n_t=200, seed=0)
print(f'A(全局,无G): dim={rA["dimension"]:.4f} rad={rA["radius"]:.4f} util={rA["utility"]:.4f}', flush=True)

# —— 协议 B：pairwise + Gaussianization（论文 B.3.1 全协议）
pairs = list(combinations(range(10), 2))
dims, rads, utils = [], [], []
gman = [gaussianize(mm) for mm in sub50]
for pa, pb in pairs:
    r = glue_analysis([gman[pa], gman[pb]], n_t=200, seed=0)
    dims.append(r["dimension"]); rads.append(r["radius"]); utils.append(r["utility"])
print(f'B(pairwise+G): dim={np.nanmean(dims):.4f}±{np.nanstd(dims):.4f} '
      f'rad={np.nanmean(rads):.4f}±{np.nanstd(rads):.4f} '
      f'util={np.nanmean(utils):.4f}±{np.nanstd(utils):.4f}  (n={len(pairs)} pairs)', flush=True)

# —— 协议 C：只 pairwise（无高斯化）——分解两个步骤的贡献
dims, rads, utils = [], [], []
for pa, pb in pairs:
    r = glue_analysis([sub50[pa], sub50[pb]], n_t=200, seed=0)
    dims.append(r["dimension"]); rads.append(r["radius"]); utils.append(r["utility"])
print(f'C(pairwise):   dim={np.nanmean(dims):.4f}±{np.nanstd(dims):.4f} '
      f'rad={np.nanmean(rads):.4f}±{np.nanstd(rads):.4f} '
      f'util={np.nanmean(utils):.4f}±{np.nanstd(utils):.4f}', flush=True)

# —— 协议 D：只 Gaussianization（全局）——
gman50 = [gaussianize(mm) for mm in sub50]
rD = glue_analysis(gman50, n_t=200, seed=0)
print(f'D(全局+G):     dim={rD["dimension"]:.4f} rad={rD["radius"]:.4f} util={rD["utility"]:.4f}', flush=True)

print('DONE', flush=True)
