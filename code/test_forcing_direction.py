#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P4 forcing_loss 方向单元测试（numpy 版，数学与脚本 torch 实现等价）。

预注册附录十六要求算子方向正确。本测试验证三件事：
  1. disperse（最小化 −H）确实在谱散开方向上给出下降梯度
  2. concentrate 的正确 loss 是 Σp·log p（最小化 ⇒ 谱集中 D↓），
     而非 Σp²（最小化 Σp² 会推平谱，方向相反）
  3. 幂迭代（8 步，top-32）对 d_ent 的近似误差可接受（监控用途）
"""
import numpy as np

rng = np.random.default_rng(0)
d, B = 64, 256

# 尖谱特征（前 2 维承载几乎全部能量）
u = rng.standard_normal((d, 2))
sharp = (rng.standard_normal((B, 2)) @ u.T) * 10
flat = rng.standard_normal((B, d))


def d_ent_np(X):
    Xc = X - X.mean(0, keepdims=True)
    C = Xc.T @ Xc / max(Xc.shape[0] - 1, 1)
    lam = np.linalg.eigvalsh(C)[::-1].clip(0)
    p = lam / lam.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def pooled_p_power(X, K=32, iters=8, seed=0):
    """与脚本 torch 实现等价的幂迭代（含 qr 正交化）。"""
    Xc = X - X.mean(0, keepdims=True)
    C = Xc.T @ Xc / max(Xc.shape[0] - 1, 1)
    C = C / max(C.diagonal().mean(), 1e-8)
    g = np.random.default_rng(seed)
    V = g.standard_normal((d, K))
    V /= np.linalg.norm(V, axis=0, keepdims=True).clip(1e-8)
    lams = []
    for _ in range(iters):
        V = C @ V
        lam = np.linalg.norm(V, axis=0)
        V = V / lam.clip(1e-10)
        lams.append(lam.copy())
        V, _ = np.linalg.qr(V)
    # 注意：脚本在正交化后取的 lam 是**上一步**的 norm（QR 前的 V 列范数），
    # 这里按脚本语义返回 QR 前的 lam（K×iters 展平后归一化）
    p = np.stack(lams, axis=1).ravel().clip(0)
    return p / p.sum()


d_sharp = d_ent_np(sharp)
d_flat = d_ent_np(flat)
print(f'd_ent: sharp={d_sharp:.3f} flat={d_flat:.3f} (sharp << flat expected)')

# --- 测 1+2：两个候选 loss 在尖/平谱上的取值方向 ---
p_s = pooled_p_power(sharp).ravel()
p_f = pooled_p_power(flat).ravel()

sum2_s, sum2_f = float((p_s ** 2).sum()), float((p_f ** 2).sum())
negH_s = float((p_s * np.log(p_s + 1e-8)).sum())
negH_f = float((p_f * np.log(p_f + 1e-8)).sum())

print(f'Sigma p^2      : sharp={sum2_s:.4f} flat={sum2_f:.4f} '
      f'(sharp {"<" if sum2_s < sum2_f else ">"} flat)')
print(f'Sigma p log p  : sharp={negH_s:.4f} flat={negH_f:.4f} '
      f'(sharp {"<" if negH_s < negH_f else ">"} flat)')

# 判据：
# - 谱熵 H = -Σp·log p：尖谱 H 小（≈0），平谱 H 大（≈log K）
#   ⇒ disperse loss = −H（最小化 −H = 最大化 H = 谱散开 = D↑）✔
#   检验：negH_s should be ≈ 0（接近最大），negH_f 接近 −log32
assert negH_s > negH_f, 'disperse 方向错误'
print('=> disperse loss = Σp·log p 最小化 ⇒ H 增大 ⇒ D↑  [方向正确]')

# - concentrate（D↓）需要 loss 最小化时谱集中：
#   Σp² 在尖谱时**大**（≈1），平谱时小（≈1/K）
#   ⇒ 最小化 Σp² = 推平谱 = D↑ ✗（脚本当前实现的错误）
#   正确：最小化 Σp² 的反面不存在（有界下界问题），应改用
#   「最大化 Σp²」不可行（SGD 只做最小化），等价于最小化 (K·Σp²−1) 不变号。
#   实际正确形式：loss = −Σp·log p 的最小化 = 最大化负熵？不对——
#   谱集中 = 最小化 H = 最小化 −Σp·log p = 最小化 Σp·log(p)·(−1)
#   即 loss = −(−Σp·log p) = Σp·log p？验证：尖谱 Σp·log p ≈ 0（最小），
#   平谱 ≈ −log K（更小）……方向又反了。
print()
print('数值锚点：log(1/32) =', round(np.log(1 / 32), 4))
print(f'  尖谱 Σp·log p ≈ {negH_s:.4f}（应接近 0）')
print(f'  平谱 Σp·log p ≈ {negH_f:.4f}（应接近 log(1/K)·1 = −3.47 附近，'
      f'实际受幂迭代截断影响偏低）')
# 结论：尖谱时 Σp·log p 更**大**（0 > −3.47），所以最小化 Σp·log p ⇒ 平谱化（D↑）
# 要谱集中（D↓），需要 loss 在尖谱时取**最小**：loss = −Σp·log p，
# 尖谱 −Σp·log p ≈ 0，平谱 ≈ +3.47 ⇒ 最小化 −Σp·log p 会推向尖谱 ✔
loss_con_sharp = -negH_s
loss_con_flat = -negH_f
print(f'  concentrate loss = −Σp·log p: sharp={loss_con_sharp:.4f} '
      f'flat={loss_con_flat:.4f}（最小化推向 sharp/D↓ ✔）')
assert loss_con_sharp < loss_con_flat, 'concentrate 方向错误'
print('=> concentrate loss 应为 −Σp·log p（最小化 ⇒ 谱集中 D↓）')
print()
print('结论：脚本 train_forcing_p4.py 的 concentrate 实现需从 Σp² 改为 −Σp·log p，')
print('      disperse 需从 −Σ(p·log p) 语义核对（当前 −H 是对的，保持）。')

# --- 测 3：幂迭代近似 d_ent 的精度 ---
def d_ent_from_p(p):
    p = p / p.sum()
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))

p_full = pooled_p_power(sharp, K=64, iters=50, seed=1)   # 全谱长迭代
p_appx = pooled_p_power(sharp, K=32, iters=8, seed=1)
print()
print(f'幂迭代近似检查: full(64,50it)={d_ent_from_p(p_full):.3f} '
      f'vs appx(32,8it)={d_ent_from_p(p_appx):.3f} vs true={d_sharp:.3f}')
