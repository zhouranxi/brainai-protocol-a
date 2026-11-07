#!/usr/bin/env python3
"""官方 example 的 GLUE 多种子方差检验：区分 MC 误差 vs 系统性偏差。

背景：我们的 glue_impl 在官方 example_checkpoint 上
  - PR 精确命中（10.2586）⇒ 特征/采样协议正确
  - GLUE 三指标有偏差（历史 50 点协议：dim +29%/rad +9%/util +18%）
  - 今日 500 点协议：dim 1.38/rad 0.537 接近，但 util=0.024 vs ref 0.19

本脚本：50 点协议（与历史验证同协议）× 10 个 downsample seed × GLUE t-seed 固定 0，
看 utility 的跨 seed 分布是否覆盖 0.19。若覆盖 ⇒ MC 噪声；若稳定偏离 ⇒ 系统性
（最可能：官方 GLUE 内部有 Gaussianization 预处理）。
"""
import sys
import numpy as np

sys.path.insert(0, '/mnt/mydisk/xinguan-brainai/code')
from glue_impl import glue_analysis, downsample_manifolds

REF = (1.3009, 0.5240, 0.1900)

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
print(f'{"seed":>4s} {"dim":>8s} {"rad":>8s} {"util":>8s}', flush=True)

dims, rads, utils = [], [], []
for s in range(10):
    sub50 = downsample_manifolds(manifolds500, n_points=50, seed=s)
    r = glue_analysis(sub50, n_t=200, seed=0)
    dims.append(r["dimension"]); rads.append(r["radius"]); utils.append(r["utility"])
    print(f'{s:4d} {r["dimension"]:8.4f} {r["radius"]:8.4f} {r["utility"]:8.4f}', flush=True)

print(f'\nmean±sd: dim {np.mean(dims):.4f}±{np.std(dims):.4f} '
      f'({100*(np.mean(dims)-REF[0])/REF[0]:+.0f}%)  '
      f'rad {np.mean(rads):.4f}±{np.std(rads):.4f} '
      f'({100*(np.mean(rads)-REF[1])/REF[1]:+.0f}%)  '
      f'util {np.mean(utils):.4f}±{np.std(utils):.4f} '
      f'({100*(np.mean(utils)-REF[2])/REF[2]:+.0f}%)', flush=True)
print('DONE', flush=True)
