# R1 精确命题检验（基于官方发布的逐模型数据）

数据：官方 `df_fig15-20.pkl`，6 个 corruption 场景 × 80 个模型。每模型含 ID acc、OOD linear probe acc、corruption acc 与全部几何标记。

## H3：task-relevant 与 task-agnostic 几何量的预测力分工

**H3**：越是与下游线性分类任务绑定的几何标记（GLUE 系），其预测力越集中在 OOD 侧；越是通用几何量，越偏向 ID 侧。

| 指标类别 | 指标 | 平均 \|r\| ID | 平均 \|r\| OOD-C | 平均 Δ(OOD−ID) | Δ>0 场景占比 |
|---|---|---|---|---|---|
| task-relevant | `dimension_id` | 0.125 | 0.648 | +0.523 | 6/6 |
| task-relevant | `center_alignment_id` | 0.103 | 0.658 | +0.555 | 6/6 |
| task-relevant | `utility_id` | 0.343 | 0.494 | +0.151 | 6/6 |
| task-relevant | `capacity_id` | 0.532 | 0.091 | -0.441 | 0/6 |
| task-relevant | `axis_alignment_id` | 0.496 | 0.443 | -0.053 | 0/6 |
| task-relevant | `center_axis_alignment_id` | 0.538 | 0.347 | -0.191 | 0/6 |
| task-relevant | `radius_id` | 0.644 | 0.638 | -0.006 | 0/6 |
| task-agnostic | `participation_ratio_id` | 0.498 | 0.003 | -0.495 | 0/6 |
| task-agnostic | `mean_distance_avg_id` | 0.751 | 0.313 | -0.438 | 0/6 |
| task-agnostic | `mean_distance_id` | 0.637 | 0.447 | -0.189 | 0/6 |
| task-agnostic | `mean_angle_id` | 0.371 | 0.430 | +0.059 | 6/6 |
| task-agnostic | `mean_covariance_id` | 0.475 | 0.414 | -0.061 | 0/6 |
| task-agnostic | `sparsity_id` | 0.171 | 0.132 | -0.038 | 0/6 |

**分组结论**

- task-relevant 标记（42 个指标-场景组合）：平均 Δ = **+0.077**，其中 18/42 个为正
- task-agnostic 几何量（36 个）：平均 Δ = **-0.194**，其中 6/36 个为正

→ **H3 成立**：task-relevant 标记的平均 Δ 比 task-agnostic 高 +0.271

## H4：ID 准确率相近的模型子集内，OOD 差异有多大？

取 ID acc 带宽 ±2.0 个百分点的模型子集，比较该子集内 OOD probe 准确率的跨度 vs 全体的跨度。

| 场景 | 子集 n | 子集内 OOD acc 跨度 | 全体 OOD acc 跨度 | 相对放大 | 子集内 ID acc 跨度 |
|---|---|---|---|---|---|
| gaussian_noise | 43 | 37.7693 | 39.2060 | **96.3%** | 3.8200 |
| fog | 43 | 37.7693 | 39.2060 | **96.3%** | 3.8200 |
| brightness | 43 | 37.7693 | 39.2060 | **96.3%** | 3.8200 |
| contrast | 43 | 37.7693 | 39.2060 | **96.3%** | 3.8200 |
| pixelate | 43 | 37.7693 | 39.2060 | **96.3%** | 3.8200 |
| jpeg_compression | 43 | 37.7693 | 39.2060 | **96.3%** | 3.8200 |

→ 子集内 OOD 跨度平均保留全体的 **96.3%**。
→ **H4 成立**：在 ID 准确率几乎相同的模型之间，OOD 性能仍有很大差异 —— 这正是几何标记能提供增量信息的空间。

## H5：在 ID 准确率相近的子集内，谁更能预测 OOD？

这是最严格的对照：只看 ID acc 相近的模型，几何标记 vs ID acc 本身，谁与 OOD 相关更强。

| 场景 | 子集 n | 指标 | \|r\| OOD (子集内) | p |
|---|---|---|---|---|
| gaussian_noise | — | `acc_test` | 0.482 | 0.0011 |
| gaussian_noise | — | `dimension_id` | 0.885 | 0.0000 |
| gaussian_noise | — | `center_alignment_id` | 0.809 | 0.0000 |
| gaussian_noise | — | `utility_id` | 0.877 | 0.0000 |
| gaussian_noise | — | `capacity_id` | 0.235 | 0.1288 |
| gaussian_noise | — | `axis_alignment_id` | 0.303 | 0.0481 |
| gaussian_noise | — | `center_axis_alignment_id` | 0.099 | 0.5285 |
| gaussian_noise | — | `radius_id` | 0.558 | 0.0001 |
| gaussian_noise | — | `participation_ratio_id` | 0.223 | 0.1508 |
| gaussian_noise | — | `mean_distance_avg_id` | 0.221 | 0.1549 |
| gaussian_noise | — | `mean_distance_id` | 0.397 | 0.0084 |
| gaussian_noise | — | `mean_angle_id` | 0.402 | 0.0075 |
| gaussian_noise | — | `mean_covariance_id` | 0.403 | 0.0074 |
| gaussian_noise | — | `sparsity_id` | 0.163 | 0.2955 |
| fog | — | `acc_test` | 0.482 | 0.0011 |
| fog | — | `dimension_id` | 0.885 | 0.0000 |
| fog | — | `center_alignment_id` | 0.809 | 0.0000 |
| fog | — | `utility_id` | 0.877 | 0.0000 |
| fog | — | `capacity_id` | 0.235 | 0.1288 |
| fog | — | `axis_alignment_id` | 0.303 | 0.0481 |
| fog | — | `center_axis_alignment_id` | 0.099 | 0.5285 |
| fog | — | `radius_id` | 0.558 | 0.0001 |
| fog | — | `participation_ratio_id` | 0.223 | 0.1508 |
| fog | — | `mean_distance_avg_id` | 0.221 | 0.1549 |
| fog | — | `mean_distance_id` | 0.397 | 0.0084 |
| fog | — | `mean_angle_id` | 0.402 | 0.0075 |
| fog | — | `mean_covariance_id` | 0.403 | 0.0074 |
| fog | — | `sparsity_id` | 0.163 | 0.2955 |
| brightness | — | `acc_test` | 0.482 | 0.0011 |
| brightness | — | `dimension_id` | 0.885 | 0.0000 |
| brightness | — | `center_alignment_id` | 0.809 | 0.0000 |
| brightness | — | `utility_id` | 0.877 | 0.0000 |
| brightness | — | `capacity_id` | 0.235 | 0.1288 |
| brightness | — | `axis_alignment_id` | 0.303 | 0.0481 |
| brightness | — | `center_axis_alignment_id` | 0.099 | 0.5285 |
| brightness | — | `radius_id` | 0.558 | 0.0001 |
| brightness | — | `participation_ratio_id` | 0.223 | 0.1508 |
| brightness | — | `mean_distance_avg_id` | 0.221 | 0.1549 |
| brightness | — | `mean_distance_id` | 0.397 | 0.0084 |
| brightness | — | `mean_angle_id` | 0.402 | 0.0075 |
| brightness | — | `mean_covariance_id` | 0.403 | 0.0074 |
| brightness | — | `sparsity_id` | 0.163 | 0.2955 |
| contrast | — | `acc_test` | 0.482 | 0.0011 |
| contrast | — | `dimension_id` | 0.885 | 0.0000 |
| contrast | — | `center_alignment_id` | 0.809 | 0.0000 |
| contrast | — | `utility_id` | 0.877 | 0.0000 |
| contrast | — | `capacity_id` | 0.235 | 0.1288 |
| contrast | — | `axis_alignment_id` | 0.303 | 0.0481 |
| contrast | — | `center_axis_alignment_id` | 0.099 | 0.5285 |
| contrast | — | `radius_id` | 0.558 | 0.0001 |
| contrast | — | `participation_ratio_id` | 0.223 | 0.1508 |
| contrast | — | `mean_distance_avg_id` | 0.221 | 0.1549 |
| contrast | — | `mean_distance_id` | 0.397 | 0.0084 |
| contrast | — | `mean_angle_id` | 0.402 | 0.0075 |
| contrast | — | `mean_covariance_id` | 0.403 | 0.0074 |
| contrast | — | `sparsity_id` | 0.163 | 0.2955 |
| pixelate | — | `acc_test` | 0.482 | 0.0011 |
| pixelate | — | `dimension_id` | 0.885 | 0.0000 |
| pixelate | — | `center_alignment_id` | 0.809 | 0.0000 |
| pixelate | — | `utility_id` | 0.877 | 0.0000 |
| pixelate | — | `capacity_id` | 0.235 | 0.1288 |
| pixelate | — | `axis_alignment_id` | 0.303 | 0.0481 |
| pixelate | — | `center_axis_alignment_id` | 0.099 | 0.5285 |
| pixelate | — | `radius_id` | 0.558 | 0.0001 |
| pixelate | — | `participation_ratio_id` | 0.223 | 0.1508 |
| pixelate | — | `mean_distance_avg_id` | 0.221 | 0.1549 |
| pixelate | — | `mean_distance_id` | 0.397 | 0.0084 |
| pixelate | — | `mean_angle_id` | 0.402 | 0.0075 |
| pixelate | — | `mean_covariance_id` | 0.403 | 0.0074 |
| pixelate | — | `sparsity_id` | 0.163 | 0.2955 |
| jpeg_compression | — | `acc_test` | 0.482 | 0.0011 |
| jpeg_compression | — | `dimension_id` | 0.885 | 0.0000 |
| jpeg_compression | — | `center_alignment_id` | 0.809 | 0.0000 |
| jpeg_compression | — | `utility_id` | 0.877 | 0.0000 |
| jpeg_compression | — | `capacity_id` | 0.235 | 0.1288 |
| jpeg_compression | — | `axis_alignment_id` | 0.303 | 0.0481 |
| jpeg_compression | — | `center_axis_alignment_id` | 0.099 | 0.5285 |
| jpeg_compression | — | `radius_id` | 0.558 | 0.0001 |
| jpeg_compression | — | `participation_ratio_id` | 0.223 | 0.1508 |
| jpeg_compression | — | `mean_distance_avg_id` | 0.221 | 0.1549 |
| jpeg_compression | — | `mean_distance_id` | 0.397 | 0.0084 |
| jpeg_compression | — | `mean_angle_id` | 0.402 | 0.0075 |
| jpeg_compression | — | `mean_covariance_id` | 0.403 | 0.0074 |
| jpeg_compression | — | `sparsity_id` | 0.163 | 0.2955 |

**各标记在子集内对 OOD 的平均 |r|**

| 标记 | 平均 \|r\| | 显著(p<0.05)比例 |
|---|---|---|
| `acc_test` | 0.482 | 6/6 |
| `dimension_id` | 0.885 | 6/6 |
| `center_alignment_id` | 0.809 | 6/6 |
| `utility_id` | 0.877 | 6/6 |
| `capacity_id` | 0.235 | 0/6 |
| `axis_alignment_id` | 0.303 | 6/6 |
| `center_axis_alignment_id` | 0.099 | 0/6 |
| `radius_id` | 0.558 | 6/6 |
| `participation_ratio_id` | 0.223 | 0/6 |
| `mean_distance_avg_id` | 0.221 | 0/6 |
| `mean_distance_id` | 0.397 | 6/6 |
| `mean_angle_id` | 0.402 | 6/6 |
| `mean_covariance_id` | 0.403 | 6/6 |
| `sparsity_id` | 0.163 | 0/6 |

→ 子集内最强预测器：**`dimension_id`**（平均 |r| = 0.885）
  vs `acc_test`（0.482）