#!/usr/bin/env python3
"""官方 example checkpoint 的 GLUE 协议变体扫描。

定位我们的 glue_impl 与官方参考值（1.3009/0.5240/0.1900）的协议差异来源。
特征缓存 /tmp/example_feat.npy（PR=10.2586 已精确命中 ⇒ 特征/采样协议正确）。
"""
import sys
import numpy as np

sys.path.insert(0, '/mnt/mydisk/xinguan-brainai/code')
from glue_impl import glue_analysis

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
manifolds1000 = [X_cls[c * 1000:(c + 1) * 1000].T for c in range(10)]

print(f'ref: dim=1.3009 rad=0.5240 util=0.1900', flush=True)

# H：每类 1000 全量
r = glue_analysis(manifolds1000, n_t=200)
print(f'H(1000pt):  dim={r["dimension"]:.4f} rad={r["radius"]:.4f} util={r["utility"]:.4f}', flush=True)

# I：500 点，t 采样 seed 方差
for s in [0, 1, 2]:
    r = glue_analysis(manifolds500, n_t=200, seed=s)
    print(f'I(500,s{s}):  dim={r["dimension"]:.4f} rad={r["radius"]:.4f} util={r["utility"]:.4f}', flush=True)

# J：nt=400 与 nt=1000（采样数收敛性）
for nt in [400, 1000]:
    r = glue_analysis(manifolds500, n_t=nt)
    print(f'J(nt={nt}): dim={r["dimension"]:.4f} rad={r["radius"]:.4f} util={r["utility"]:.4f}', flush=True)

print('DONE', flush=True)
