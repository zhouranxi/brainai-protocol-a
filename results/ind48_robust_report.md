# 自建 48 组：Protocol A 复核 + C4 逐族 + 剔除 lr=1.0 敏感性

生成：2026-10-27 · band=±2pp · bootstrap B=2000 (seed=42)

## 背景

原稿 §4.7 只报了汇总残差化系数（基线 −0.439 / −0.338），掩盖了逐 lr 族的符号翻转。
按论文自身的 C4（要求所有组符号一致），基线在本数据上**不通过 C4**。
同时 lr=1.0 族并非收敛子群（OOD 7.1–14.7 vs 其余 13.7–32.4），属训练失败区。

## 全量 48 组

n=48 → ID 匹配后 n=46；族：[0.125, 0.25, 0.5, 1.0]

| 预测器 | CIFAR-100 残差化 r | Tiny-ImageNet 残差化 r |
|---|---|---|
| Participation ratio (per-class) | +0.701 | +0.707 |
| Local smoothness | +0.832 | +0.792 |
| Numerical rank | +0.870 | +0.853 |
| Effective dim (PCA) | +0.715 | +0.708 |
| Class separability | -0.763 | -0.722 |
| **ID accuracy (baseline)** | **-0.439** | **-0.338** |

**C4 逐族符号（基线，残差化）：**

| lr 族 | n | CIFAR-100 r | Tiny r |
|---|---|---|---|
| 0.125 | 12 | -0.802 | -0.739 |
| 0.25 | 11 | -0.603 | -0.134 |
| 0.5 | 12 | -0.330 | -0.186 |
| 1.0 | 11 | +0.694 ⚠ 正号 | +0.415 |

→ C4 符号一致：**否**

**Bootstrap 95% CI（剔除 lr=1.0 敏感性口径见下）**

| 预测器 | 目标 | 95% CI | 均值 | P(r>0) |
|---|---|---|---|---|
| Participation ratio (per-class) | CIFAR-100 | [+0.532, +0.810] | +0.693 | 100.0% |
| Participation ratio (per-class) | Tiny-ImageNet | [+0.510, +0.822] | +0.696 | 100.0% |
| Local smoothness | CIFAR-100 | [+0.741, +0.899] | +0.832 | 100.0% |
| Local smoothness | Tiny-ImageNet | [+0.671, +0.875] | +0.793 | 100.0% |
| Numerical rank | CIFAR-100 | [+0.800, +0.923] | +0.870 | 100.0% |
| Numerical rank | Tiny-ImageNet | [+0.776, +0.919] | +0.854 | 100.0% |
| Effective dim (PCA) | CIFAR-100 | [+0.565, +0.830] | +0.715 | 100.0% |
| Effective dim (PCA) | Tiny-ImageNet | [+0.546, +0.838] | +0.709 | 100.0% |
| Class separability | CIFAR-100 | [-0.847, -0.654] | -0.762 | 0.0% |
| Class separability | Tiny-ImageNet | [-0.820, -0.581] | -0.721 | 0.0% |
| ID accuracy (baseline) | CIFAR-100 | [-0.668, -0.173] | -0.448 | 0.2% |
| ID accuracy (baseline) | Tiny-ImageNet | [-0.581, -0.063] | -0.346 | 1.1% |

## 收敛族 36 组（剔除 lr=1.0）

n=36 → ID 匹配后 n=34；族：[0.125, 0.25, 0.5]

| 预测器 | CIFAR-100 残差化 r | Tiny-ImageNet 残差化 r |
|---|---|---|
| Participation ratio (per-class) | +0.701 | +0.723 |
| Local smoothness | +0.843 | +0.820 |
| Numerical rank | +0.892 | +0.884 |
| Effective dim (PCA) | +0.729 | +0.738 |
| Class separability | -0.794 | -0.763 |
| **ID accuracy (baseline)** | **-0.658** | **-0.513** |

**C4 逐族符号（基线，残差化）：**

| lr 族 | n | CIFAR-100 r | Tiny r |
|---|---|---|---|
| 0.125 | 11 | -0.966 | -0.943 |
| 0.25 | 11 | -0.603 | -0.134 |
| 0.5 | 12 | -0.330 | -0.186 |

→ C4 符号一致：**是**

**Bootstrap 95% CI（剔除 lr=1.0 敏感性口径见下）**

| 预测器 | 目标 | 95% CI | 均值 | P(r>0) |
|---|---|---|---|---|
| Participation ratio (per-class) | CIFAR-100 | [+0.516, +0.813] | +0.691 | 100.0% |
| Participation ratio (per-class) | Tiny-ImageNet | [+0.506, +0.838] | +0.707 | 100.0% |
| Local smoothness | CIFAR-100 | [+0.716, +0.915] | +0.833 | 100.0% |
| Local smoothness | Tiny-ImageNet | [+0.653, +0.893] | +0.802 | 100.0% |
| Numerical rank | CIFAR-100 | [+0.819, +0.943] | +0.890 | 100.0% |
| Numerical rank | Tiny-ImageNet | [+0.796, +0.938] | +0.879 | 100.0% |
| Effective dim (PCA) | CIFAR-100 | [+0.563, +0.846] | +0.727 | 100.0% |
| Effective dim (PCA) | Tiny-ImageNet | [+0.554, +0.857] | +0.728 | 100.0% |
| Class separability | CIFAR-100 | [-0.880, -0.649] | -0.784 | 0.0% |
| Class separability | Tiny-ImageNet | [-0.858, -0.573] | -0.746 | 0.0% |
| ID accuracy (baseline) | CIFAR-100 | [-0.803, -0.420] | -0.636 | 0.0% |
| ID accuracy (baseline) | Tiny-ImageNet | [-0.695, -0.217] | -0.486 | 0.1% |

## 结论

1. **全量 48 组**：基线残差化 −0.439/−0.338，但逐族符号为 −0.80/−0.60/−0.33/**+0.69**，按 C4 不通过——原稿「负号复现」的表述不成立。
2. **收敛族 36 组**：基线负号更强且**三族全负**（−0.966/−0.603/−0.330），C4 通过，bootstrap CI 全部排除零（P(r>0)≈0%）。
3. 几何量在两个口径下都稳定为正（numerical rank +0.870→+0.892），bootstrap 稳健率 100%。
4. → 论文表述应从「基线负号复现」改为「**基线符号不稳定（全量）/ 稳定为负（收敛族）**」，这更契合「基线不是稳定锚」的核心论点。

## 修订要求

- §4.7 表格增列 C4 逐族符号表（附录亦可）。
- Limitations 新增一条：lr=1.0 族为低 OOD 水平子群，已做剔除敏感性分析。
- 摘要中「baseline instability replicates in direction」措辞改为符号稳定性表述。
