#!/usr/bin/env python3
"""B.3.1 协议假设检验：pairwise GLUE + Gaussianization + 全局 P 聚合。

假设：官方 glue_analysis(sub_manifolds) 内部对 P=10 个 manifold 做
  ① pairwise 分解（C(10,2)=45 对）
  ② 每对做 Gaussianization + GLUE
  ③ dimension 聚合用 (1/P_global)·E[b]（不是 1/2）

验证：官方 example 参考值 dim=1.3009 rad=0.5240 util=0.1900
"""
import sys
import numpy as np
from itertools import combinations

sys.path.insert(0, '/mnt/mydisk/xinguan-brainai/code')
from glue_impl import glue_analysis, downsample_manifolds

REF = (1.3009, 0.5240, 0.1900)


def gaussianize(manifold, eps=1e-12):
    from scipy.stats import norm
    N, M = manifold.shape
    out = np.empty_like(manifold, dtype=np.float64)
    for i in range(N):
        v = manifold[i]
        order = np.argsort(v, kind='stable')
        ranks = np.empty(M)
        sv = v[order]
        i0 = 0
        while i0 < M:
            j = i0
            while j + 1 < M and sv[j + 1] - sv[j] <= eps:
                j += 1
            mid = (i0 + j) / 2.0 + 1.0
            ranks[order[i0:j + 1]] = mid
            i0 = j + 1
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
sub50 = downsample_manifolds(manifolds500, n_points=50, seed=0)
gman = [gaussianize(mm) for mm in sub50]

P_GLOBAL = 10
pairs = list(combinations(range(P_GLOBAL), 2))
print(f'ref: dim=1.3009 rad=0.5240 util=0.1900', flush=True)

# —— E：pairwise + G，dim = (1/P_global)·mean_pair(b) ——
dims, rads, utils = [], [], []
for pa, pb in pairs:
    r = glue_analysis([gman[pa], gman[pb]], n_t=200, seed=0)
    dims.append(r["dimension"]); rads.append(r["radius"]); utils.append(r["utility"])
# glue_analysis 对 P=2 返回 dim=(1/2)·b_mean，所以 b_mean = 2·dim
b_per_pair = [2 * d for d in dims]
dim_E = np.mean(b_per_pair) / P_GLOBAL
print(f'E(pairwise+G, /P10): dim={dim_E:.4f}', flush=True)

# —— F：不除 P，直接平均 pairwise dimension（=(1/2)·b̄）——
dim_F = np.mean(dims)
print(f'F(pairwise+G, /2):   dim={dim_F:.4f}', flush=True)

# —— G：utility 与 radius 的 pairwise 平均 ——
print(f'G(pairwise+G): rad={np.mean(rads):.4f} util={np.mean(utils):.4f}', flush=True)

print('DONE', flush=True)
