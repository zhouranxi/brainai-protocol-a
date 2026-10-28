# 用官方发布数据直接检验 R1 命题

数据源：官方仓库 `data/df_fig15-20.pkl`（6 个 corruption 场景，每场景 80 个模型 = 5 架构 × 4 lr × ... ）

> 说明：这是**作者自己算好的逐模型测量值**，含 ID 准确率、OOD linear probe 准确率、以及在 ID 数据上计算的全部几何指标。
> 因此可以在**完全相同的数据**上检验「几何标记的预测力是否分布偏移专属」。

场景：gaussian_noise, fog, brightness, contrast, pixelate, jpeg_compression

## H1：几何标记对 OOD-C 的预测力 vs 对 ID 的预测力

OD-C = CIFAR-100 linear probe 准确率（`acc_lp_ood`）；ID = CIFAR-10 test 准确率（`acc_test`）

| 标记 | 场景 | \|r\| ID | p(ID) | \|r\| OOD-C | p(OOD-C) | Δ\|r\| | 判定 |
|---|---|---|---|---|---|---|---|---|
| `dimension_id` | gaussian_noise | 0.125 | 0.2686 | 0.648 | 0.0000 | +0.523 | **OOD 更强** |
| `participation_ratio_id` | gaussian_noise | 0.498 | 0.0000 | 0.003 | 0.9799 | -0.495 | ID 更强 |
| `utility_id` | gaussian_noise | 0.343 | 0.0018 | 0.494 | 0.0000 | +0.151 | **OOD 更强** |
| `center_alignment_id` | gaussian_noise | 0.103 | 0.3634 | 0.658 | 0.0000 | +0.555 | **OOD 更强** |
| `sparsity_id` | gaussian_noise | 0.171 | 0.1301 | 0.132 | 0.2420 | -0.038 | 相当 |
| `mean_covariance_id` | gaussian_noise | 0.475 | 0.0000 | 0.414 | 0.0001 | -0.061 | ID 更强 |
| `mean_distance_avg_id` | gaussian_noise | 0.751 | 0.0000 | 0.313 | 0.0047 | -0.438 | ID 更强 |
| `radius_id` | gaussian_noise | 0.644 | 0.0000 | 0.638 | 0.0000 | -0.006 | 相当 |
| `axis_alignment_id` | gaussian_noise | 0.496 | 0.0000 | 0.443 | 0.0000 | -0.053 | ID 更强 |
| `center_axis_alignment_id` | gaussian_noise | 0.538 | 0.0000 | 0.347 | 0.0016 | -0.191 | ID 更强 |
| `mean_distance_id` | gaussian_noise | 0.637 | 0.0000 | 0.447 | 0.0000 | -0.189 | ID 更强 |
| `mean_angle_id` | gaussian_noise | 0.371 | 0.0007 | 0.430 | 0.0001 | +0.059 | **OOD 更强** |
| `capacity_id` | gaussian_noise | 0.532 | 0.0000 | 0.091 | 0.4227 | -0.441 | ID 更强 |
| `dimension_id` | fog | 0.125 | 0.2686 | 0.648 | 0.0000 | +0.523 | **OOD 更强** |
| `participation_ratio_id` | fog | 0.498 | 0.0000 | 0.003 | 0.9799 | -0.495 | ID 更强 |
| `utility_id` | fog | 0.343 | 0.0018 | 0.494 | 0.0000 | +0.151 | **OOD 更强** |
| `center_alignment_id` | fog | 0.103 | 0.3634 | 0.658 | 0.0000 | +0.555 | **OOD 更强** |
| `sparsity_id` | fog | 0.171 | 0.1301 | 0.132 | 0.2420 | -0.038 | 相当 |
| `mean_covariance_id` | fog | 0.475 | 0.0000 | 0.414 | 0.0001 | -0.061 | ID 更强 |
| `mean_distance_avg_id` | fog | 0.751 | 0.0000 | 0.313 | 0.0047 | -0.438 | ID 更强 |
| `radius_id` | fog | 0.644 | 0.0000 | 0.638 | 0.0000 | -0.006 | 相当 |
| `axis_alignment_id` | fog | 0.496 | 0.0000 | 0.443 | 0.0000 | -0.053 | ID 更强 |
| `center_axis_alignment_id` | fog | 0.538 | 0.0000 | 0.347 | 0.0016 | -0.191 | ID 更强 |
| `mean_distance_id` | fog | 0.637 | 0.0000 | 0.447 | 0.0000 | -0.189 | ID 更强 |
| `mean_angle_id` | fog | 0.371 | 0.0007 | 0.430 | 0.0001 | +0.059 | **OOD 更强** |
| `capacity_id` | fog | 0.532 | 0.0000 | 0.091 | 0.4227 | -0.441 | ID 更强 |
| `dimension_id` | brightness | 0.125 | 0.2686 | 0.648 | 0.0000 | +0.523 | **OOD 更强** |
| `participation_ratio_id` | brightness | 0.498 | 0.0000 | 0.003 | 0.9799 | -0.495 | ID 更强 |
| `utility_id` | brightness | 0.343 | 0.0018 | 0.494 | 0.0000 | +0.151 | **OOD 更强** |
| `center_alignment_id` | brightness | 0.103 | 0.3634 | 0.658 | 0.0000 | +0.555 | **OOD 更强** |
| `sparsity_id` | brightness | 0.171 | 0.1301 | 0.132 | 0.2420 | -0.038 | 相当 |
| `mean_covariance_id` | brightness | 0.475 | 0.0000 | 0.414 | 0.0001 | -0.061 | ID 更强 |
| `mean_distance_avg_id` | brightness | 0.751 | 0.0000 | 0.313 | 0.0047 | -0.438 | ID 更强 |
| `radius_id` | brightness | 0.644 | 0.0000 | 0.638 | 0.0000 | -0.006 | 相当 |
| `axis_alignment_id` | brightness | 0.496 | 0.0000 | 0.443 | 0.0000 | -0.053 | ID 更强 |
| `center_axis_alignment_id` | brightness | 0.538 | 0.0000 | 0.347 | 0.0016 | -0.191 | ID 更强 |
| `mean_distance_id` | brightness | 0.637 | 0.0000 | 0.447 | 0.0000 | -0.189 | ID 更强 |
| `mean_angle_id` | brightness | 0.371 | 0.0007 | 0.430 | 0.0001 | +0.059 | **OOD 更强** |
| `capacity_id` | brightness | 0.532 | 0.0000 | 0.091 | 0.4227 | -0.441 | ID 更强 |
| `dimension_id` | contrast | 0.125 | 0.2686 | 0.648 | 0.0000 | +0.523 | **OOD 更强** |
| `participation_ratio_id` | contrast | 0.498 | 0.0000 | 0.003 | 0.9799 | -0.495 | ID 更强 |
| `utility_id` | contrast | 0.343 | 0.0018 | 0.494 | 0.0000 | +0.151 | **OOD 更强** |
| `center_alignment_id` | contrast | 0.103 | 0.3634 | 0.658 | 0.0000 | +0.555 | **OOD 更强** |
| `sparsity_id` | contrast | 0.171 | 0.1301 | 0.132 | 0.2420 | -0.038 | 相当 |
| `mean_covariance_id` | contrast | 0.475 | 0.0000 | 0.414 | 0.0001 | -0.061 | ID 更强 |
| `mean_distance_avg_id` | contrast | 0.751 | 0.0000 | 0.313 | 0.0047 | -0.438 | ID 更强 |
| `radius_id` | contrast | 0.644 | 0.0000 | 0.638 | 0.0000 | -0.006 | 相当 |
| `axis_alignment_id` | contrast | 0.496 | 0.0000 | 0.443 | 0.0000 | -0.053 | ID 更强 |
| `center_axis_alignment_id` | contrast | 0.538 | 0.0000 | 0.347 | 0.0016 | -0.191 | ID 更强 |
| `mean_distance_id` | contrast | 0.637 | 0.0000 | 0.447 | 0.0000 | -0.189 | ID 更强 |
| `mean_angle_id` | contrast | 0.371 | 0.0007 | 0.430 | 0.0001 | +0.059 | **OOD 更强** |
| `capacity_id` | contrast | 0.532 | 0.0000 | 0.091 | 0.4227 | -0.441 | ID 更强 |
| `dimension_id` | pixelate | 0.125 | 0.2686 | 0.648 | 0.0000 | +0.523 | **OOD 更强** |
| `participation_ratio_id` | pixelate | 0.498 | 0.0000 | 0.003 | 0.9799 | -0.495 | ID 更强 |
| `utility_id` | pixelate | 0.343 | 0.0018 | 0.494 | 0.0000 | +0.151 | **OOD 更强** |
| `center_alignment_id` | pixelate | 0.103 | 0.3634 | 0.658 | 0.0000 | +0.555 | **OOD 更强** |
| `sparsity_id` | pixelate | 0.171 | 0.1301 | 0.132 | 0.2420 | -0.038 | 相当 |
| `mean_covariance_id` | pixelate | 0.475 | 0.0000 | 0.414 | 0.0001 | -0.061 | ID 更强 |
| `mean_distance_avg_id` | pixelate | 0.751 | 0.0000 | 0.313 | 0.0047 | -0.438 | ID 更强 |
| `radius_id` | pixelate | 0.644 | 0.0000 | 0.638 | 0.0000 | -0.006 | 相当 |
| `axis_alignment_id` | pixelate | 0.496 | 0.0000 | 0.443 | 0.0000 | -0.053 | ID 更强 |
| `center_axis_alignment_id` | pixelate | 0.538 | 0.0000 | 0.347 | 0.0016 | -0.191 | ID 更强 |
| `mean_distance_id` | pixelate | 0.637 | 0.0000 | 0.447 | 0.0000 | -0.189 | ID 更强 |
| `mean_angle_id` | pixelate | 0.371 | 0.0007 | 0.430 | 0.0001 | +0.059 | **OOD 更强** |
| `capacity_id` | pixelate | 0.532 | 0.0000 | 0.091 | 0.4227 | -0.441 | ID 更强 |
| `dimension_id` | jpeg_compression | 0.125 | 0.2686 | 0.648 | 0.0000 | +0.523 | **OOD 更强** |
| `participation_ratio_id` | jpeg_compression | 0.498 | 0.0000 | 0.003 | 0.9799 | -0.495 | ID 更强 |
| `utility_id` | jpeg_compression | 0.343 | 0.0018 | 0.494 | 0.0000 | +0.151 | **OOD 更强** |
| `center_alignment_id` | jpeg_compression | 0.103 | 0.3634 | 0.658 | 0.0000 | +0.555 | **OOD 更强** |
| `sparsity_id` | jpeg_compression | 0.171 | 0.1301 | 0.132 | 0.2420 | -0.038 | 相当 |
| `mean_covariance_id` | jpeg_compression | 0.475 | 0.0000 | 0.414 | 0.0001 | -0.061 | ID 更强 |
| `mean_distance_avg_id` | jpeg_compression | 0.751 | 0.0000 | 0.313 | 0.0047 | -0.438 | ID 更强 |
| `radius_id` | jpeg_compression | 0.644 | 0.0000 | 0.638 | 0.0000 | -0.006 | 相当 |
| `axis_alignment_id` | jpeg_compression | 0.496 | 0.0000 | 0.443 | 0.0000 | -0.053 | ID 更强 |
| `center_axis_alignment_id` | jpeg_compression | 0.538 | 0.0000 | 0.347 | 0.0016 | -0.191 | ID 更强 |
| `mean_distance_id` | jpeg_compression | 0.637 | 0.0000 | 0.447 | 0.0000 | -0.189 | ID 更强 |
| `mean_angle_id` | jpeg_compression | 0.371 | 0.0007 | 0.430 | 0.0001 | +0.059 | **OOD 更强** |
| `capacity_id` | jpeg_compression | 0.532 | 0.0000 | 0.091 | 0.4227 | -0.441 | ID 更强 |

### 跨场景汇总

| 标记 | 场景数 | OOD 更强的场景数 | 平均 \|r\| ID | 平均 \|r\| OOD-C | 平均 Δ |
|---|---|---|---|---|---|
| `dimension_id` | 6 | 6/6 | 0.125 | 0.648 | +0.523 |
| `participation_ratio_id` | 6 | 0/6 | 0.498 | 0.003 | -0.495 |
| `utility_id` | 6 | 6/6 | 0.343 | 0.494 | +0.151 |
| `center_alignment_id` | 6 | 6/6 | 0.103 | 0.658 | +0.555 |
| `sparsity_id` | 6 | 0/6 | 0.171 | 0.132 | -0.038 |
| `mean_covariance_id` | 6 | 0/6 | 0.475 | 0.414 | -0.061 |
| `mean_distance_avg_id` | 6 | 0/6 | 0.751 | 0.313 | -0.438 |
| `radius_id` | 6 | 0/6 | 0.644 | 0.638 | -0.006 |
| `axis_alignment_id` | 6 | 0/6 | 0.496 | 0.443 | -0.053 |
| `center_axis_alignment_id` | 6 | 0/6 | 0.538 | 0.347 | -0.191 |
| `mean_distance_id` | 6 | 0/6 | 0.637 | 0.447 | -0.189 |
| `mean_angle_id` | 6 | 6/6 | 0.371 | 0.430 | +0.059 |
| `capacity_id` | 6 | 0/6 | 0.532 | 0.091 | -0.441 |

**结论：4/13 个指标在多数场景下，几何标记对 OOD 的预测力强于对 ID。**

→ H1 不成立。

## H2：corruption 偏移下，ID accuracy 是否最强预测器？

预测目标 = `acc_test_c`（corruption 后的准确率），预测变量取自 ID 数据

| 场景 | 最强预测器 | \|r\| | acc_test 的 \|r\| | 排名 |
|---|---|---|---|---|
| gaussian_noise | `mean_angle_id` | 0.666 | 0.182 | 8/14 |
| fog | `acc_test` | 0.987 | 0.987 | 1/14 |
| brightness | `acc_test` | 0.994 | 0.994 | 1/14 |
| contrast | `acc_test` | 0.965 | 0.965 | 1/14 |
| pixelate | `center_alignment_id` | 0.451 | 0.122 | 11/14 |
| jpeg_compression | `center_alignment_id` | 0.382 | 0.122 | 12/14 |

**结论：acc_test 在 3/6 个场景中排进前 2。**
→ H2 部分成立（与论文结论有出入，需讨论）。

## 与官方 Figure 4 真值对照（ResNet18/SGD，CIFAR-100）

| 官方指标 | 官方 r | 官方 p |
|---|---|---|
| `dimension_id` | +0.799 | 1.03e-11 |
| `participation_ratio_id` | +0.796 | 1.30e-11 |
| `center_alignment_id` | +0.790 | 2.37e-11 |
| `utility_id` | +0.740 | 1.92e-09 |
| `acc_train` | +0.734 | 2.94e-09 |
| `sparsity_id` | +0.630 | 1.59e-06 |
| `entropy` | -0.616 | 3.10e-06 |
| `numerical_rank_id` | +0.592 | 9.36e-06 |
| `mean_distance_avg_id` | +0.487 | 4.55e-04 |
| `axis_alignment_id` | -0.459 | 1.02e-03 |
| `center_axis_alignment_id` | -0.430 | 2.28e-03 |
| `mean_covariance_id` | +0.414 | 3.41e-03 |
| `auroc` | +0.304 | 3.58e-02 |
| `radius_id` | -0.292 | 4.37e-02 |
| `acc_test` | +0.173 | 2.39e-01 |
| `mean_distance_id` | -0.122 | 4.08e-01 |
| `nc1_id` | +0.108 | 4.65e-01 |
| `mean_angle_id` | -0.075 | 6.13e-01 |

> 官方 `acc_test` 的 r = **+0.173（不显著）**，而 `dimension_id` / `participation_ratio_id` 均 > +0.79 —— 这正是「几何标记比 ID accuracy 更能预测 OOD」的官方证据。