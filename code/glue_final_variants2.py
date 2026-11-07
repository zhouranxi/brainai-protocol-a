#!/usr/bin/env python3
"""kappa 细扫 + t 归一化（修复递归 bug）。

K(kappa) 初扫：kappa=1.0 时 util=0.1862 已接近 ref 0.19，方向正确。
这里细扫 kappa ∈ {1.5, 2.0, 3.0, 5.0} + t 单位球归一化。

ref: dim=1.3009 rad=0.5240 util=0.1900
"""
import sys
import numpy as np

sys.path.insert(0, '/mnt/mydisk/xinguan-brainai/code')
from glue_impl import glue_analysis, downsample_manifolds

_orig_default_rng = np.random.default_rng


class UnitSphereRng:
    """产生单位球面 t 的 rng（构造时用原版 default_rng，避免递归）。"""

    def __init__(self, base_seed=0):
        self.rng = _orig_default_rng(base_seed)

    def standard_normal(self, n):
        v = self.rng.standard_normal(n)
        return v / np.linalg.norm(v)

    def choice(self, *a, **k):
        return self.rng.choice(*a, **k)


data = np.load('/tmp/example_feat.npy', allow_pickle=True).item()
X_all, y = data['X'], data['y']
order = np.argsort(y, kind='stable')
X_cls = X_all[order]
rng = _orig_default_rng(42)
sub_idx = np.concatenate([rng.choice(1000, 500, replace=False) + c * 1000
                          for c in range(10)])
X_sub = X_cls[sub_idx]
manifolds500 = [X_sub[c * 500:(c + 1) * 500].T for c in range(10)]
sub50 = downsample_manifolds(manifolds500, n_points=50, seed=0)

print(f'ref: dim=1.3009 rad=0.5240 util=0.1900', flush=True)

# kappa 细扫
for kap in [1.5, 2.0, 3.0, 5.0]:
    r = glue_analysis(sub50, n_t=200, seed=0, kappa=kap)
    print(f'K(kappa={kap}): dim={r["dimension"]:.4f} rad={r["radius"]:.4f} util={r["utility"]:.4f}', flush=True)

# t 单位球归一化（kappa=0 与 kappa=1.0 两档）
for kap in [0.0, 1.0]:
    np.random.default_rng = lambda seed=0, _k=kap: UnitSphereRng(seed)
    try:
        rL = glue_analysis(sub50, n_t=200, seed=0, kappa=_k)
    finally:
        np.random.default_rng = _orig_default_rng
    print(f'L(t归一,k={kap}): dim={rL["dimension"]:.4f} rad={rL["radius"]:.4f} util={rL["utility"]:.4f}', flush=True)

print('DONE', flush=True)
