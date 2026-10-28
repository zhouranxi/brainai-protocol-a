# R1 分析：几何标记的预测力是否分布偏移专属

- 模型数 n = 8
- 评估协议（对齐 ICLR 2026 论文 Section 3.1）：
  - ID：模型原分类头在 CIFAR-10 test 上的准确率
  - OOD-C：**linear probe**（冻结特征，在 CIFAR-100 train 上训新头，test 评估）
  - OOD-K：CIFAR-10C（severity 5），标签空间不变 → 直接用原分类头
- 几何标记**只从 ID 数据**计算（预后性 prognostic 的定义）

## 逐模型明细

| model | ID acc | OOD-C probe | OOD-K | PR(per-class) | PR(global) | NC1 | num.rank |
|---|---|---|---|---|---|---|---|
| `aug_only_s2` | 0.8532 | 0.2213 | 0.6419 | 6.13 | 9.36 | 0.5807 | 377 |
| `best_s2` | 0.8529 | 0.1914 | 0.6405 | 5.24 | 10.49 | 0.3775 | 343 |
| `noise20_aug_s2` | 0.8515 | 0.2196 | 0.6385 | 6.24 | 9.39 | 0.5910 | 384 |
| `plain_s2` | 0.7808 | 0.1587 | 0.6295 | 5.72 | 9.40 | 0.7327 | 397 |
| `plain_ls_s2` | 0.7762 | 0.1123 | 0.6253 | 3.72 | 9.41 | 0.6508 | 273 |
| `lr_low_s2` | 0.7477 | 0.2890 | 0.6014 | 19.42 | 22.34 | 1.1720 | 495 |
| `overfit_s2` | 0.7353 | 0.2213 | 0.6061 | 11.33 | 15.85 | 1.1062 | 476 |
| `wd0_s2` | 0.7347 | 0.2277 | 0.5995 | 12.93 | 14.38 | 1.2539 | 482 |

## 相关性矩阵（几何标记 × 预测目标）

| 标记 | 类型 | ID (CIFAR-10) | OOD-C (CIFAR-100 probe) | OOD-K (CIFAR-10C) | |Δr| OOD-C − ID |
|---|---|---|---|---|---|
| `id_pr_perclass` | 非平凡 | -0.640 | +0.824* | -0.828* | +0.184 |
| `id_pr_global` | 非平凡 | -0.657 | +0.746* | -0.819* | +0.088 |
| `id_nc1` | 非平凡 | -0.897*** | +0.527 | -0.968*** | -0.371 |
| `id_numerical_rank` | 非平凡 | -0.594 | +0.818* | -0.750* | +0.224 |
| `id_effective_dim_pca` | 非平凡 | -0.665 | +0.749* | -0.831* | +0.084 |
| `id_local_smoothness` | 非平凡 | -0.688 | +0.814* | -0.862*** | +0.126 |
| `id_class_separability` | 平凡 | +0.833* | -0.651 | +0.959*** | -0.182 |

显著性：`*` p<0.05，`***` p<0.01

## H1 检验：几何标记对 OOD-C 的预测力是否强于对 ID？

- `id_pr_perclass`: |r|_ID = 0.640 (p=0.0874)  vs  |r|_OOD-C = 0.824 (p=0.0119)  → **OOD-C 更强**
- `id_pr_global`: |r|_ID = 0.657 (p=0.0766)  vs  |r|_OOD-C = 0.746 (p=0.0337)  → **OOD-C 更强**
- `id_nc1`: |r|_ID = 0.897 (p=0.0025)  vs  |r|_OOD-C = 0.527 (p=0.1800)  → ID 更强
- `id_numerical_rank`: |r|_ID = 0.594 (p=0.1209)  vs  |r|_OOD-C = 0.818 (p=0.0131)  → **OOD-C 更强**
- `id_effective_dim_pca`: |r|_ID = 0.665 (p=0.0721)  vs  |r|_OOD-C = 0.749 (p=0.0324)  → **OOD-C 更强**
- `id_local_smoothness`: |r|_ID = 0.688 (p=0.0591)  vs  |r|_OOD-C = 0.814 (p=0.0139)  → **OOD-C 更强**

**结论：5/6 个非平凡标记对 OOD-C 的预测力强于 ID。**

→ 支持 H1：**几何标记的预测力是分布偏移专属的**。

## H2 检验：corruption 偏移下是否 ID accuracy 最强？

按 |r| 排序（对 OOD-K 的预测力）：

- `id_nc1`: |r| = 0.968 
- `id_class_separability`: |r| = 0.959 平凡
- `id_local_smoothness`: |r| = 0.862 
- `id_effective_dim_pca`: |r| = 0.831 
- `id_pr_perclass`: |r| = 0.828 
- `id_pr_global`: |r| = 0.819 
- `id_numerical_rank`: |r| = 0.750 

最强：id_nc1。
→ 未完全支持 H2。

## 预测准确率（论文 §4 口径：几何标记 vs ID accuracy 选模型）

- 按 ID accuracy 选最好的模型 → 其 OOD-C 排名中位数：

| 预测器 | 选中模型的 OOD-C acc | OOD-C 排名 (1=最好) | 命中率 |
|---|---|---|---|
| ID accuracy | 0.2213 | 3/8 | — |
| `id_pr_perclass` | 0.2890 | 1/8 | ✓ 选中 OOD 最好 |
| `id_pr_global` | 0.2890 | 1/8 | ✓ 选中 OOD 最好 |
| `id_nc1` | 0.1914 | 6/8 |  |
| `id_numerical_rank` | 0.2890 | 1/8 | ✓ 选中 OOD 最好 |
| `id_effective_dim_pca` | 0.2890 | 1/8 | ✓ 选中 OOD 最好 |
| `id_local_smoothness` | 0.1123 | 8/8 |  |
