# 偏移类型分解：双口径对照（pooled vs Protocol A）

生成：2026-10-27 · band=±2pp · 数据：official_df_fig15-20.pkl（80 模型 × 6 场景）

背景：原稿只报 pooled Spearman ρ，但整节放在 Protocol A 框架下叙述。
本表并列两个口径：pooled = 全 80 模型相关；Protocol A = ID 匹配（±2pp）+ 架构残差化后相关。

### gaussian_noise（n=43 after matching）

| 指标 | pooled ρ | Protocol A ρ |
|---|---|---|
| ID accuracy | +0.182 | -0.087 |
| participation ratio | +0.208 | +0.279 |
| effective dimension | +0.398 | +0.228 |
| effective utility | +0.269 | +0.221 |
| effective radius | -0.540 | -0.096 |
| center alignment | -0.008 | +0.112 |
| axis alignment | -0.565 | -0.020 |
| capacity | +0.013 | -0.003 |
| mean distance | +0.072 | +0.052 |
| mean angle | -0.666 | -0.135 |
| mean covariance | -0.028 | -0.008 |
| sparsity | -0.181 | +0.297 |

最佳（pooled）= mean angle；最佳（Protocol A）= sparsity；ID acc 秩：pooled 7/12，Protocol A 8/12

### fog（n=43 after matching）

| 指标 | pooled ρ | Protocol A ρ |
|---|---|---|
| ID accuracy | +0.987 | +0.918 |
| participation ratio | -0.516 | -0.334 |
| effective dimension | -0.168 | -0.453 |
| effective utility | -0.373 | -0.450 |
| effective radius | -0.587 | -0.049 |
| center alignment | -0.113 | -0.642 |
| axis alignment | -0.430 | -0.029 |
| capacity | +0.544 | +0.397 |
| mean distance | -0.642 | -0.756 |
| mean angle | -0.298 | +0.496 |
| mean covariance | -0.478 | -0.676 |
| sparsity | -0.154 | -0.322 |

最佳（pooled）= ID accuracy；最佳（Protocol A）= ID accuracy；ID acc 秩：pooled 1/12，Protocol A 1/12

### brightness（n=43 after matching）

| 指标 | pooled ρ | Protocol A ρ |
|---|---|---|
| ID accuracy | +0.994 | +0.968 |
| participation ratio | -0.518 | -0.402 |
| effective dimension | -0.167 | -0.384 |
| effective utility | -0.375 | -0.395 |
| effective radius | -0.603 | -0.120 |
| center alignment | -0.136 | -0.604 |
| axis alignment | -0.457 | -0.014 |
| capacity | +0.547 | +0.389 |
| mean distance | -0.620 | -0.686 |
| mean angle | -0.333 | +0.400 |
| mean covariance | -0.452 | -0.569 |
| sparsity | -0.158 | -0.202 |

最佳（pooled）= ID accuracy；最佳（Protocol A）= ID accuracy；ID acc 秩：pooled 1/12，Protocol A 1/12

### contrast（n=43 after matching）

| 指标 | pooled ρ | Protocol A ρ |
|---|---|---|
| ID accuracy | +0.965 | +0.772 |
| participation ratio | -0.511 | -0.220 |
| effective dimension | -0.008 | -0.289 |
| effective utility | -0.226 | -0.265 |
| effective radius | -0.668 | +0.080 |
| center alignment | -0.004 | -0.467 |
| axis alignment | -0.525 | -0.015 |
| capacity | +0.462 | +0.266 |
| mean distance | -0.629 | -0.563 |
| mean angle | -0.429 | +0.358 |
| mean covariance | -0.474 | -0.501 |
| sparsity | -0.114 | -0.327 |

最佳（pooled）= ID accuracy；最佳（Protocol A）= ID accuracy；ID acc 秩：pooled 1/12，Protocol A 1/12

### pixelate（n=43 after matching）

| 指标 | pooled ρ | Protocol A ρ |
|---|---|---|
| ID accuracy | -0.122 | +0.215 |
| participation ratio | +0.247 | +0.238 |
| effective dimension | -0.370 | +0.237 |
| effective utility | -0.349 | +0.161 |
| effective radius | +0.123 | -0.369 |
| center alignment | -0.451 | +0.084 |
| axis alignment | +0.053 | -0.305 |
| capacity | +0.194 | +0.078 |
| mean distance | +0.346 | -0.095 |
| mean angle | +0.181 | +0.089 |
| mean covariance | +0.302 | -0.025 |
| sparsity | -0.020 | +0.515 |

最佳（pooled）= center alignment；最佳（Protocol A）= sparsity；ID acc 秩：pooled 10/12，Protocol A 6/12

### jpeg_compression（n=43 after matching）

| 指标 | pooled ρ | Protocol A ρ |
|---|---|---|
| ID accuracy | +0.122 | -0.010 |
| participation ratio | +0.192 | +0.323 |
| effective dimension | -0.068 | +0.441 |
| effective utility | -0.135 | +0.405 |
| effective radius | -0.265 | -0.210 |
| center alignment | -0.382 | +0.106 |
| axis alignment | -0.309 | -0.227 |
| capacity | +0.241 | +0.022 |
| mean distance | +0.348 | -0.004 |
| mean angle | -0.310 | -0.015 |
| mean covariance | +0.314 | +0.052 |
| sparsity | -0.119 | +0.602 |

最佳（pooled）= center alignment；最佳（Protocol A）= sparsity；ID acc 秩：pooled 10/12，Protocol A 11/12

## 结论

- 两口径在**定性结论上完全一致**：photometric/structural（fog/brightness/contrast）下 ID 准确率均为第 1 名；noise/compression 下均失效。
- 但幅度与最优标记有差异：contrast 由 +0.965 降至 +0.772；pixelate 基线由 −0.12 变 +0.22，最优标记由 center alignment 变为 sparsity。
- → 偏移类型结论必须在两个口径下分别陈述；只报 pooled 是在测混杂，不是测现象。
