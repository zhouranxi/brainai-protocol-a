#!/usr/bin/env python3
"""最后两个高概率协议变体：t 归一化 & kappa margin。

排除史：
  A(全局,无G):  dim=1.6781 rad=0.4745 util=0.1555（+29%/−9%/−18%）
  B-G(pairwise 各聚合): 全排除（dim 3.9–46.6）
  D(按类+G):    util 命中但 dim=16.8 崩
  H(pooled白化)/I(pooled高斯)/J(全量高斯): dim 4.3–28.7 全排除
  ⇒ 高斯化作用于特征都会让 dim 飙升，方向不对

剩余高概率假设：
  K：t 采样后归一化到单位球面（||t||=1）——GLUE 理论中 t 是"随机方向"
  L：kappa margin > 0（QP 的软间隔）
ref: dim=1.3009 rad=0.5240 util=0.1900
"""
import sys
import numpy as np

sys.path.insert(0, '/mnt/mydisk/xinguan-brainai/code')
from glue_impl import glue_analysis, downsample_manifolds

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

print(f'ref: dim=1.3009 rad=0.5240 util=0.1900', flush=True)

# K：kappa 扫描（0.1 / 0.5 / 1.0）
for kap in [0.1, 0.5, 1.0]:
    r = glue_analysis(sub50, n_t=200, seed=0, kappa=kap)
    print(f'K(kappa={kap}): dim={r["dimension"]:.4f} rad={r["radius"]:.4f} util={r["utility"]:.4f}', flush=True)

# L：t 归一化——需要 monkey-patch rng。直接改 glue_impl 不合适，
# 用一个 wrapper：复制 glue_analysis 的代码改 t 生成（这里用简单方法：
# 传一个预归一化的自定义 rng 类）
class UnitSphereRng:
    """产生归一化 t 的 rng 替身（只实现 standard_normal）。"""
    def __init__(self, base_seed=0):
        self.rng = np.random.default_rng(base_seed)
    def standard_normal(self, n):
        v = self.rng.standard_normal(n)
        return v / np.linalg.norm(v)
    def choice(self, *a, **k):
        return self.rng.choice(*a, **k)

import glue_impl
_orig = glue_impl.glue_analysis
# 临时替换 numpy default_rng 构造（glue_impl 内部 rng = np.random.default_rng(seed)）
_orig_default_rng = np.random.default_rng
np.random.default_rng = UnitSphereRng
try:
    rL = glue_analysis(sub50, n_t=200, seed=0)
finally:
    np.random.default_rng = _orig_default_rng
print(f'L(t归一化):   dim={rL["dimension"]:.4f} rad={rL["radius"]:.4f} util={rL["utility"]:.4f}', flush=True)

print('DONE', flush=True)
