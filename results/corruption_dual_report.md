# 偏移类型分解：双口径对照（pooled vs Protocol A）× 双统计量（Pearson / Spearman）

生成：2026-11-03 · band=±2pp · 数据：official_df_fig15-20.pkl（80 模型 × 6 场景）

背景：原稿只报 pooled Spearman ρ，但整节放在 Protocol A 框架下叙述。
本表并列两个口径：pooled = 全 80 模型相关；Protocol A = ID 匹配（±2pp）+ 架构残差化后相关。

统计量说明：2026-11-03 修正 —— 原实现用 `np.corrcoef`（**Pearson**）却把数值标成 ρ（Spearman）。
下表同时给出 **Pearson r** 与 **Spearman ρ**（`r_pooled`/`r_Protocol A` 为 Pearson，`ρ_pooled`/`ρ_Protocol A` 为 Spearman）。论文主口径为 Spearman。

### gaussian_noise（n=43 after matching）

| 指标 | r_pooled (Pearson) | r_Protocol A (Pearson) | ρ_pooled (Spearman) | ρ_Protocol A (Spearman) |
|---|---|---|---|---|
| ID accuracy | +0.182 | -0.087 | +0.185 | -0.109 |
| participation ratio | +0.208 | +0.279 | +0.070 | +0.229 |
| effective dimension | +0.398 | +0.228 | +0.357 | +0.216 |
| effective utility | +0.269 | +0.221 | +0.250 | +0.212 |
| effective radius | -0.540 | -0.096 | -0.454 | -0.041 |
| center alignment | -0.008 | +0.112 | +0.133 | +0.083 |
| axis alignment | -0.565 | -0.020 | -0.451 | +0.021 |
| capacity | +0.013 | -0.003 | -0.020 | -0.051 |
| mean distance | +0.072 | +0.052 | -0.024 | +0.084 |
| mean angle | -0.666 | -0.135 | -0.632 | -0.067 |
| mean covariance | -0.028 | -0.008 | -0.050 | +0.112 |
| sparsity | -0.181 | +0.297 | -0.146 | +0.299 |

最佳（pooled, Spearman）= mean angle；最佳（Protocol A, Spearman）= sparsity；ID acc 秩（Spearman）：pooled 6/12，Protocol A 6/12

（参考，Pearson 口径：最佳 pooled = mean angle，最佳 Protocol A = sparsity；ID acc 秩 pooled 7/12，Protocol A 8/12）

### fog（n=43 after matching）

| 指标 | r_pooled (Pearson) | r_Protocol A (Pearson) | ρ_pooled (Spearman) | ρ_Protocol A (Spearman) |
|---|---|---|---|---|
| ID accuracy | +0.987 | +0.918 | +0.990 | +0.934 |
| participation ratio | -0.516 | -0.334 | -0.496 | -0.516 |
| effective dimension | -0.168 | -0.453 | -0.144 | -0.509 |
| effective utility | -0.373 | -0.450 | -0.358 | -0.475 |
| effective radius | -0.587 | -0.049 | -0.689 | -0.125 |
| center alignment | -0.113 | -0.642 | -0.027 | -0.534 |
| axis alignment | -0.430 | -0.029 | -0.572 | -0.018 |
| capacity | +0.544 | +0.397 | +0.508 | +0.412 |
| mean distance | -0.642 | -0.756 | -0.727 | -0.747 |
| mean angle | -0.298 | +0.496 | -0.192 | +0.260 |
| mean covariance | -0.478 | -0.676 | -0.611 | -0.609 |
| sparsity | -0.154 | -0.322 | -0.015 | -0.478 |

最佳（pooled, Spearman）= ID accuracy；最佳（Protocol A, Spearman）= ID accuracy；ID acc 秩（Spearman）：pooled 1/12，Protocol A 1/12

（参考，Pearson 口径：最佳 pooled = ID accuracy，最佳 Protocol A = ID accuracy；ID acc 秩 pooled 1/12，Protocol A 1/12）

### brightness（n=43 after matching）

| 指标 | r_pooled (Pearson) | r_Protocol A (Pearson) | ρ_pooled (Spearman) | ρ_Protocol A (Spearman) |
|---|---|---|---|---|
| ID accuracy | +0.994 | +0.968 | +0.994 | +0.973 |
| participation ratio | -0.518 | -0.402 | -0.480 | -0.511 |
| effective dimension | -0.167 | -0.384 | -0.131 | -0.465 |
| effective utility | -0.375 | -0.395 | -0.343 | -0.441 |
| effective radius | -0.603 | -0.120 | -0.694 | -0.126 |
| center alignment | -0.136 | -0.604 | -0.033 | -0.538 |
| axis alignment | -0.457 | -0.014 | -0.587 | -0.013 |
| capacity | +0.547 | +0.389 | +0.504 | +0.395 |
| mean distance | -0.620 | -0.686 | -0.725 | -0.723 |
| mean angle | -0.333 | +0.400 | -0.216 | +0.191 |
| mean covariance | -0.452 | -0.569 | -0.612 | -0.553 |
| sparsity | -0.158 | -0.202 | -0.032 | -0.372 |

最佳（pooled, Spearman）= ID accuracy；最佳（Protocol A, Spearman）= ID accuracy；ID acc 秩（Spearman）：pooled 1/12，Protocol A 1/12

（参考，Pearson 口径：最佳 pooled = ID accuracy，最佳 Protocol A = ID accuracy；ID acc 秩 pooled 1/12，Protocol A 1/12）

### contrast（n=43 after matching）

| 指标 | r_pooled (Pearson) | r_Protocol A (Pearson) | ρ_pooled (Spearman) | ρ_Protocol A (Spearman) |
|---|---|---|---|---|
| ID accuracy | +0.965 | +0.772 | +0.975 | +0.816 |
| participation ratio | -0.511 | -0.220 | -0.541 | -0.370 |
| effective dimension | -0.008 | -0.289 | -0.055 | -0.422 |
| effective utility | -0.226 | -0.265 | -0.255 | -0.384 |
| effective radius | -0.668 | +0.080 | -0.694 | +0.051 |
| center alignment | -0.004 | -0.467 | +0.054 | -0.458 |
| axis alignment | -0.525 | -0.015 | -0.608 | +0.039 |
| capacity | +0.462 | +0.266 | +0.456 | +0.262 |
| mean distance | -0.629 | -0.563 | -0.716 | -0.605 |
| mean angle | -0.429 | +0.358 | -0.286 | +0.214 |
| mean covariance | -0.474 | -0.501 | -0.606 | -0.574 |
| sparsity | -0.114 | -0.327 | +0.058 | -0.489 |

最佳（pooled, Spearman）= ID accuracy；最佳（Protocol A, Spearman）= ID accuracy；ID acc 秩（Spearman）：pooled 1/12，Protocol A 1/12

（参考，Pearson 口径：最佳 pooled = ID accuracy，最佳 Protocol A = ID accuracy；ID acc 秩 pooled 1/12，Protocol A 1/12）

### pixelate（n=43 after matching）

| 指标 | r_pooled (Pearson) | r_Protocol A (Pearson) | ρ_pooled (Spearman) | ρ_Protocol A (Spearman) |
|---|---|---|---|---|
| ID accuracy | -0.122 | +0.215 | -0.128 | +0.191 |
| participation ratio | +0.247 | +0.238 | +0.309 | +0.017 |
| effective dimension | -0.370 | +0.237 | -0.358 | -0.074 |
| effective utility | -0.349 | +0.161 | -0.347 | -0.114 |
| effective radius | +0.123 | -0.369 | +0.124 | -0.300 |
| center alignment | -0.451 | +0.084 | -0.445 | +0.064 |
| axis alignment | +0.053 | -0.305 | +0.110 | -0.120 |
| capacity | +0.194 | +0.078 | +0.159 | -0.064 |
| mean distance | +0.346 | -0.095 | +0.315 | -0.131 |
| mean angle | +0.181 | +0.089 | +0.165 | +0.037 |
| mean covariance | +0.302 | -0.025 | +0.332 | +0.006 |
| sparsity | -0.020 | +0.515 | -0.274 | +0.106 |

最佳（pooled, Spearman）= center alignment；最佳（Protocol A, Spearman）= effective radius；ID acc 秩（Spearman）：pooled 10/12，Protocol A 2/12

（参考，Pearson 口径：最佳 pooled = center alignment，最佳 Protocol A = sparsity；ID acc 秩 pooled 10/12，Protocol A 6/12）

### jpeg_compression（n=43 after matching）

| 指标 | r_pooled (Pearson) | r_Protocol A (Pearson) | ρ_pooled (Spearman) | ρ_Protocol A (Spearman) |
|---|---|---|---|---|
| ID accuracy | +0.122 | -0.010 | +0.087 | -0.055 |
| participation ratio | +0.192 | +0.323 | +0.250 | +0.161 |
| effective dimension | -0.068 | +0.441 | -0.038 | +0.231 |
| effective utility | -0.135 | +0.405 | -0.109 | +0.262 |
| effective radius | -0.265 | -0.210 | -0.251 | -0.077 |
| center alignment | -0.382 | +0.106 | -0.316 | +0.025 |
| axis alignment | -0.309 | -0.227 | -0.243 | -0.003 |
| capacity | +0.241 | +0.022 | +0.177 | -0.016 |
| mean distance | +0.348 | -0.004 | +0.191 | -0.001 |
| mean angle | -0.310 | -0.015 | -0.293 | -0.028 |
| mean covariance | +0.314 | +0.052 | +0.201 | +0.174 |
| sparsity | -0.119 | +0.602 | -0.351 | +0.216 |

最佳（pooled, Spearman）= sparsity；最佳（Protocol A, Spearman）= effective utility；ID acc 秩（Spearman）：pooled 11/12，Protocol A 7/12

（参考，Pearson 口径：最佳 pooled = center alignment，最佳 Protocol A = sparsity；ID acc 秩 pooled 10/12，Protocol A 11/12）

## 结论（Spearman 口径为主）

- gaussian_noise：最佳标记 pooled=mean angle，Protocol A=sparsity；ID acc 秩 pooled 6/12，Protocol A 6/12
- fog：最佳标记 pooled=ID accuracy，Protocol A=ID accuracy；ID acc 秩 pooled 1/12，Protocol A 1/12
- brightness：最佳标记 pooled=ID accuracy，Protocol A=ID accuracy；ID acc 秩 pooled 1/12，Protocol A 1/12
- contrast：最佳标记 pooled=ID accuracy，Protocol A=ID accuracy；ID acc 秩 pooled 1/12，Protocol A 1/12
- pixelate：最佳标记 pooled=center alignment，Protocol A=effective radius；ID acc 秩 pooled 10/12，Protocol A 2/12
- jpeg_compression：最佳标记 pooled=sparsity，Protocol A=effective utility；ID acc 秩 pooled 11/12，Protocol A 7/12

- Spearman 下 pooled 与 Protocol A 的最佳标记不一致的场景：gaussian_noise, pixelate, jpeg_compression。

### Pearson vs Spearman 差异（统计量口径本身的影响）

| 场景 | 最佳标记 pooled（Pearson→Spearman） | 最佳标记 Protocol A（Pearson→Spearman） | ID acc 秩 pooled | ID acc 秩 Protocol A |
|---|---|---|---|---|
| gaussian_noise | mean angle → mean angle | sparsity → sparsity | 7/12 → 6/12 | 8/12 → 6/12 |
| fog | ID accuracy → ID accuracy | ID accuracy → ID accuracy | 1/12 → 1/12 | 1/12 → 1/12 |
| brightness | ID accuracy → ID accuracy | ID accuracy → ID accuracy | 1/12 → 1/12 | 1/12 → 1/12 |
| contrast | ID accuracy → ID accuracy | ID accuracy → ID accuracy | 1/12 → 1/12 | 1/12 → 1/12 |
| pixelate | center alignment → center alignment | sparsity → effective radius | 10/12 → 10/12 | 6/12 → 2/12 |
| jpeg_compression | center alignment → sparsity | sparsity → effective utility | 10/12 → 11/12 | 11/12 → 7/12 |

- → 偏移类型结论必须在两个口径下分别陈述；只报 pooled 是在测混杂，不是测现象。
- → 同时必须标明统计量：Pearson 与 Spearman 在 noise/compression 场景可给出不同的最佳标记。
