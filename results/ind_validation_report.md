# 独立验证报告：自建 48 组模型上的受控协议复现

模型：48 组 resnet18 / CIFAR-10，4 lr × 4 wd × 3 seed
指标：公开可算几何量（participation_ratio / numerical_rank / NC1 / mean_* / sparsity / local_smoothness / effective_dim_pca）
OOD：CIFAR-100 linear probe（冻结特征，OOD 上训新头，与官方协议一致）

> **本层的目的**：验证「ID 受控 + 架构/超参去混淆」这个**协议**在完全独立训练的数据上是否依然成立，而不依赖于官方发布的测量值。

## V1 全样本：各公开几何量的预测力

| 指标 | r(ID) | rho(ID) | r(OOD) | rho(OOD) |
|---|---|---|---|---|
| `id_class_separability` | +0.143 | +0.327 | -0.805 | -0.698 |
| `id_effective_dim_pca` | -0.321 | -0.166 | +0.805 | +0.866 |
| `id_local_smoothness` | -0.090 | -0.220 | +0.899 | +0.866 |
| `id_nc1` | -0.423 | -0.689 | -0.296 | +0.037 |
| `id_numerical_rank` | -0.044 | +0.102 | +0.944 | +0.972 |
| `id_pr_global` | +0.003 | -0.128 | +0.879 | +0.836 |
| `id_pr_perclass` | -0.074 | +0.149 | +0.951 | +0.946 |

## V2 ID 受控子集（|acc − mean| ≤ 2 个百分点）

子集 n = 46 / 48
子集内 ID 跨度 = 3.34 个百分点；OOD 跨度 = 22.12 个百分点（全体 25.30）

残差化分别对 **lr 族**与 **lr×wd 组合**做（自建实验只有单一架构，故以超参族替代架构作为混杂源）。

| 指标 | 原始 r | 残差化(lr) r | p | 残差化(lr×wd) r | Spearman rho | 稳健? |
|---|---|---|---|---|---|---|
| `id_class_separability` | -0.767 | -0.763 | 0.0000 | -0.371 | -0.657 | ✓ |
| `id_effective_dim_pca` | +0.790 | +0.715 | 0.0000 | +0.509 | +0.849 | ✓ |
| `id_local_smoothness` | +0.889 | +0.832 | 0.0000 | +0.685 | +0.848 | ✓ |
| `id_nc1` | -0.149 | +0.121 | 0.4243 | -0.111 | +0.050 | ✗ |
| `id_numerical_rank` | +0.958 | +0.870 | 0.0000 | +0.728 | +0.976 | ✓ |
| `id_pr_global` | +0.860 | +0.748 | 0.0000 | +0.449 | +0.814 | ✓ |
| `id_pr_perclass` | +0.942 | +0.701 | 0.0000 | +0.604 | +0.940 | ✓ |

**基线 `id_acc`（ID 准确率）**：原始 r=-0.037 → 残差化 r=-0.439 (p=0.0023)，Spearman rho=+0.041

## V3 逐超参族分解（符号一致性检验）

| 指标 | 0.125 | 0.25 | 0.5 | 1.0 | 一致? |
|---|---|---|---|---|---|
| `id_class_separability` | -0.93 | -0.60 | -0.47 | -0.88 | ✓ |
| `id_effective_dim_pca` | +0.89 | +0.68 | +0.45 | +0.94 | ✓ |
| `id_local_smoothness` | +0.92 | +0.73 | +0.74 | +0.96 | ✓ |
| `id_nc1` | +0.86 | +0.57 | +0.53 | -0.02 | ✗ |
| `id_numerical_rank` | +0.97 | +0.71 | +0.83 | +0.92 | ✓ |
| `id_pr_global` | +0.90 | +0.64 | +0.30 | +0.92 | ✓ |
| `id_pr_perclass` | +0.96 | +0.47 | -0.35 | +0.83 | ✗ |
| `id_acc` | -0.80 | -0.60 | -0.33 | +0.69 | ✗ |

**留一 lr 族（用其余族拟合效应，套到被留出的族上）**

| 指标 | 0.125 | 0.25 | 0.5 | 1.0 |
|---|---|---|---|
| `id_acc` | +nan | +nan | +nan | +nan |

## V4 带宽敏感性

| 指标 | b=1pp | b=2pp | b=3pp | b=5pp | b=8pp |
|---|---|---|---|---|---|
| `id_class_separability` | -0.611 | -0.763 | -0.776 | -0.805 | -0.805 |
| `id_effective_dim_pca` | +0.542 | +0.715 | +0.696 | +0.755 | +0.755 |
| `id_local_smoothness` | +0.730 | +0.832 | +0.848 | +0.883 | +0.883 |
| `id_nc1` | -0.009 | +0.121 | -0.294 | -0.235 | -0.235 |
| `id_numerical_rank` | +0.830 | +0.870 | +0.834 | +0.878 | +0.878 |
| `id_pr_global` | +0.596 | +0.748 | +0.775 | +0.797 | +0.797 |
| `id_pr_perclass` | +0.487 | +0.701 | +0.725 | +0.806 | +0.806 |
| `id_acc` | -0.355 | -0.439 | -0.250 | -0.457 | -0.457 |

## V5 corruption 场景细分

| 场景 | 最强预测器 | |rho| | `acc_test` 排名 | `acc_test` rho |
|---|---|---|---|---|
| brightness | `id_acc` | 0.726 | 1/8 | +0.726 |
| contrast | `id_class_separability` | 0.195 | 6/8 | +0.124 |
| defocus_blur | `id_acc` | 0.524 | 1/8 | +0.524 |
| elastic_transform | `id_acc` | 0.581 | 1/8 | +0.581 |
| fog | `id_acc` | 0.685 | 1/8 | +0.685 |
| frost | `id_acc` | 0.653 | 1/8 | +0.653 |
| gaussian_blur | `id_acc` | 0.371 | 1/8 | +0.371 |
| gaussian_noise | `id_numerical_rank` | 0.756 | 7/8 | +0.078 |
| glass_blur | `id_numerical_rank` | 0.680 | 7/8 | +0.089 |
| impulse_noise | `id_local_smoothness` | 0.595 | 7/8 | -0.252 |
| jpeg_compression | `id_pr_perclass` | 0.412 | 4/8 | +0.323 |
| motion_blur | `id_acc` | 0.688 | 1/8 | +0.688 |
| pixelate | `id_pr_global` | 0.397 | 7/8 | +0.138 |
| saturate | `id_acc` | 0.695 | 1/8 | +0.695 |
| shot_noise | `id_numerical_rank` | 0.754 | 7/8 | +0.108 |
| snow | `id_acc` | 0.723 | 1/8 | +0.723 |
| spatter | `id_acc` | 0.480 | 1/8 | +0.480 |
| speckle_noise | `id_numerical_rank` | 0.720 | 7/8 | +0.182 |
| zoom_blur | `id_acc` | 0.637 | 1/8 | +0.637 |

## V6 与官方数据的一致性对照

同一协议、同一指标族，两套完全独立的数据：

| 指标 | 官方 80 模型（架构残差化 r） | 自建 48 组（lr 族残差化 r） | 一致? |
|---|---|---|---|
| `class_separability_id` | — | -0.763 | (官方无对应列) |
| `effective_dim_pca_id` | — | +0.715 | (官方无对应列) |
| `local_smoothness_id` | — | +0.832 | (官方无对应列) |
| `nc1_id` | — | +0.121 | (官方无对应列) |
| `numerical_rank_id` | — | +0.870 | (官方无对应列) |
| `participation_ratio_id` | +0.659 | +0.701 | ✓ 同向 |

> 官方数据中的 `dimension_id`/`utility_id` 来自 GLUE 私有实现，本层不含；可比的是 participation_ratio / mean_* 系列。
