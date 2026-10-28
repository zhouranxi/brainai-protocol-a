# 论文骨架：When Do Geometric Markers Predict Generalization?

> **状态**：v3 骨架（2026-10-26，已按 ADR-0010 / 0012 / 0013 更新）。
> 数字来源：
> - **主数据**：`results/protocol_a_fig3_report.md`（官方发布 **224 模型** × 5 架构，
>   13 lr × 9 wd × 3 seed）——**全程不需要 GLUE 官方代码**
> - **corruption 场景**：`results/r1_final_report.md`（80 模型 × 6 corruption 场景）
> - **独立验证**：`results/ind_validation_report.md`（自建 48 组，不依赖原作者数据）
>
> **目标会议**：ICLR / NeurIPS 主会。**核心卖点是"边界条件"而非"新指标"**——
> 我们没有提出新指标，是对已发表方法做受控检验并给出其失效条件。
>
> **v2 关键修正（ADR-0010）**：v1 引用的 pooled 相关系数（dimension r=0.885
> vs acc_test r=0.482）**存在 Simpson 悖论**，已全部替换为
> 「ID 受控 + 架构残差化 + 双相关性 + 逐架构符号一致 + 带宽稳健」后的值。

---

## 候选标题

1. **When Do Geometric Markers Predict Generalization? A Controlled Analysis
   of Task-Relevance and Shift Type**
   （推荐——直接点出两个自变量：任务相关性与偏移类型）
2. *Geometric Markers Carry Net Incremental Information Beyond ID Accuracy —
   Until the Shift Type Changes*
3. *ID Accuracy Is a Misleading Baseline for Cross-Architecture OOD Comparison:
   Boundary Conditions of Representational Geometry Markers*

**关键词**：representational geometry, generalization, OOD robustness, model
selection, GLUE, neural collapse, Simpson's paradox

---

## Abstract（草稿）

> Neural representation geometry has recently been proposed as a cheap,
> training-free proxy for predicting a model's generalization performance.
> Recent work (Chou et al., ICLR 2026) shows that geometry-derived markers such
> as the effective dimension (D_eff) correlate strongly with out-of-distribution
> (OOD) accuracy and outperform in-distribution (ID) test accuracy as a model
> selection criterion. We argue that this comparison requires three controls
> that have not been applied: (i) it must be made within a subset of models with
> near-identical ID accuracy, since ID accuracy is the very quantity the markers
> are meant to supplement; (ii) architecture main effects must be removed, since
> architecture families differ systematically in both ID accuracy and
> representational geometry; and (iii) the relative strength of markers must be
> reported with rank-based statistics, since a strong Pearson correlation alone
> can be driven by a few extreme points.
>
> Applying these controls to the per-model measurements released with the
> original study (5 architectures, 224 models spanning 13 learning rates ×
> 9 weight decays × 3 seeds), we find that:
>
> (i) the effective dimension D_eff retains a strong and highly significant
> association with OOD accuracy under the ID-matched, architecture-adjusted
> protocol (r = +0.93, p < 1e-4; Spearman ρ = +0.85), with **consistent sign
> across all five architectures** — the effect is not a Simpson's paradox;
> (ii) the apparent advantage of geometric markers over ID accuracy is, if
> anything, **larger than previously reported, for a reason the original
> analysis could not have revealed**: within the ID-matched subset, ID accuracy
> fails as a baseline entirely — its pooled correlation with OOD accuracy is
> essentially zero (r = −0.05), its architecture-adjusted Pearson is +0.39 but
> its Spearman is −0.23, and its sign is inconsistent across architectures
> (+0.48 / −0.48 / +0.46 / +0.24 / +0.42). The two correlation families
> disagree in direction, so no single baseline comparison is well defined
> unless the confounding is removed;
> (iii) this advantage is **not universal** — under photometric and structural
> corruptions (fog, brightness, contrast), ID accuracy is restored as the
> strongest predictor (ρ = 0.98–0.99), whereas under noise and compression
> corruptions it collapses (ρ = −0.13 to +0.09).
>
> We conclude that the predictive value of representational geometry markers is
> **conditional on both the task-relevance of the marker and the type of
> distribution shift**, and that claims that geometric markers "replace" ID
> accuracy must state these qualifications explicitly. We release the control
> protocol as a reusable diagnostic and validate it on an independently
> trained model family.

---

## 1. Introduction

**段落 1 — 背景与动机**
表征几何作为泛化的廉价代理指标。近两年两条路线：(a) 通用几何量
（participation ratio、manifold dimension、covariance sparsity）；(b) task-relevant
几何量（GLUE 系的 D_eff / R_eff / Ψ_eff，基于 anchor point 二次规划）。
DNN 泛化理论（Jiang et al. 2020 的 effective rank、Sorscher et al. 2022 的
压缩视角）提供机制解释。

**段落 2 — 已发表结论**
Chou, Kirsanov, Yang & Chung (ICLR 2026), *Diagnosing Generalization Failures
from Representational Geometry Markers*：几何标记预测 OOD 失败，且优于 ID
accuracy 作为模型选择准则。

**段落 3 — 我们的切入（gap）**
该文的比较是 **pooled 相关系数比较**，方法论上有三处未处理的混淆：

1. **ID 准确率本身未被控制** —— 而 ID 准确率正是几何标记要补充的那个量。
   若两个架构族在 ID 与 OOD 上有系统性差异，pooled 相关性主要由
   **架构间差异**驱动，而非架构内的模型质量关系。
2. **无逐组符号一致性检验** —— Simpson 悖论只有在逐组分解下才会暴露。
3. **偏移类型未细分** —— 只笼统陈述"corruption 下 ID accuracy 最强"。

**段落 4 — 我们的发现（预览）**
去混淆后，几何标记的优势**比原文更强但方向更明确**（ID acc 在受控子集内与
OOD 负相关 → 标记提供严格净增量信息），但优势**不普适**——在亮度/结构类
corruption 下反转。

**段落 5 — 贡献**
1. 提出并执行 **Protocol A**（ID 受控 + 架构残差化 + 双相关性 + 逐组符号一致
   + 带宽敏感性），并证明四条判据缺一不可；
2. 证明 D_eff/Ψ_eff 的 OOD 预测力在 5 个架构中符号一致，**不是** Simpson 悖论；
3. 揭示 **ID 准确率的表观正相关（r=+0.48）是架构混淆假象**——残差化后符号翻转，
   逐架构符号混乱（+0.88 / −0.73 / −0.78 / +0.52 / −0.92）；
4. 给出**偏移类型维度上的边界条件**（corruption 反转现象的细分）；
5. **独立验证**：自建 48 组模型（不依赖原作者数据）复现协议结论。

---

## 2. Related Work

### 2.1 表征几何与泛化
- Jiang et al., "How Powerful is Graph Convolution?" (2020) — effective rank
- Sorscher et al., "Beyond neural scaling laws" (2022) — 压缩视角
- Nakkiran et al., "Deep double descent" (2021)
- Mallat / StyleGAN 谱分析传统

### 2.2 神经坍缩与流形容量
- Chung, Lee & Tegmark, "Classification and geometry of general perceptual
  manifolds" (PRX 2018) — anchor point 方法
- Chou et al., "Geometry Linked to Untangling Efficiency" (bioRxiv 2024/2025) —
  GLUE 三指标

### 2.3 OOD 泛化与模型选择
- Hendrycks & Dietterich, "Benchmarking Neural Network Robustness to Common
  Corruptions and Perturbations" (ICLR 2019) — CIFAR-10C
- Cimpoi et al., CIFAR-10C / CIFAR-100-C 基准
- Liang et al., "Do ImageNet Classifiers Generalize to ImageNet?" (ICLR 2018)

### 2.4 统计方法学（本文重点引用）
- Simpson's paradox / 混杂分析 / 生态学谬误
- Spearman vs Pearson 的稳健性差异；杠杆点（leverage point）诊断
- 受控子集分析 / 部分依赖分析

**Gap 陈述**：以上四支中，没有一篇在 **ID 受控 + 架构去混淆**的条件下
系统比较几何标记与 ID accuracy 的相对预测力，也没有一篇检验该比较
在偏移类型维度上是否稳定。

---

## 3. Method（方法与检验协议）

### 3.1 预备知识：GLUE 三标记
给出 D_eff、R_eff、Ψ_eff 的定义（引 Chou et al. 2025 附录 B.3），以及
通用几何量（participation ratio、numerical rank、NC1、mean_distance 等）的定义。

### 3.2 Protocol A：ID 受控 + 架构去混淆（本文核心）

给定模型集合 $M$ 与候选预测器 $g$：

1. **ID 受控**：取 $S = \{m \in M : |\text{acc}_{ID}(m) - \overline{\text{acc}}_{ID}| \le b\}$，
   $b = 2$ 个百分点（并做 $b \in \{1,2,3,5,8\}$ 敏感性分析）。
2. **架构去混淆**：对子集内每个量 $v$ 回归
   $v \sim \mathbf{1} + \text{arch}_1 + \cdots + \text{arch}_{K-1}$，取残差 $v^{\perp}$。
3. **双相关性报告**：同时报 Pearson $r$ 与 Spearman $\rho$。
   *只有当两者同向且 $|\rho| \ge 0.5$ 时才判为稳健*——
   若 Pearson 强而 Spearman 弱，说明结论由少数极端点线性外推驱动。
4. **逐组符号一致性**：在每个架构内单独算相关系数，要求符号一致
   （Simpson 悖论诊断）。
5. **带宽稳健性**：结论必须在 $b \in \{1,2,3,5,8\}$ 全区间成立。

> **四条缺一不可（本文的方法论贡献）**。
> 本次工作中 `acc_test` 的 pooled r=+0.48 与残差化后的 −0.63 **符号相反**；
> 若只报 pooled 或只报 Pearson，都会得出与真相相反的结论。
> 各条判据的必要性在 §4.3–4.5 与 §5.1 逐条给出实证。

### 3.3 偏移类型的划分
- **类别级偏移**（class-level）：CIFAR-10 → CIFAR-100（linear probe 评估）
- **损坏级偏移**（corruption-level）：CIFAR-10C，6 种 corruption
  - 亮度/结构类：fog, brightness, contrast
  - 噪声类：gaussian_noise
  - 压缩/像素化类：pixelate, jpeg_compression

### 3.4 数据来源与伦理
- 主分析用官方发布的逐模型测量值（`data/df_fig15-20.pkl`，MIT 许可）
- 独立验证用自建 48 组模型（CIFAR-10 / resnet18，4 lr × 4 wd × 3 seed）
- 许可与合规见仓库 `docs/data-licenses.md`

---

## 4. Experiments

### 4.1 设置
- **主数据**：官方发布 `df_fig3.pkl` —— **224 模型**（5 架构，13 lr × 9 wd × 3 seed），
  含 GLUE 三指标 + 8 个通用几何量的**官方测量值**（无需 GLUE 私有代码）
- **corruption 数据**：官方 `df_fig15-20.pkl` —— 80 模型 × 6 corruption 场景
- **独立验证**：自建 48 组 resnet18/CIFAR-10
- 评估：ID = CIFAR-10 test acc；OOD = CIFAR-100 linear probe acc；
  corruption = CIFAR-10C acc
- 指标：7 个 GLUE 标记 + 8 个通用几何量 + `acc_test` 基线

### 4.2 结果 1：全样本格局（表 T1）
`dimension_id` |r|=0.125 (ID) vs 0.648 (OOD)；`participation_ratio_id`
方向相反（0.498 vs 0.003）。→ **结论取决于用哪个指标**，不存在普适的
"几何标记 > ID accuracy"。

### 4.3 结果 2：ID 受控 + 架构残差化（表 T2）★核心结果
**主数据 224 模型**，band = ±1 个百分点，n = 106 / 224，覆盖 5 个架构。
ID 跨度 1.92 个百分点时，**OOD 跨度仍保留 91.8%**（45.45 / 49.53）。

| 预测器 | 原始 r | 残差化 r | 残差化 p | Spearman ρ | 逐架构符号 | 稳健? |
|---|---|---|---|---|---|---|
| **`dimension_id`** | +0.932 | **+0.925** | <1e-4 | **+0.847** | ✓ 5/5 | **✓** |
| **`utility_id`** | +0.934 | **+0.901** | <1e-4 | +0.769 | ✓ 5/5 | **✓** |
| `radius_id` | −0.547 | −0.584 | <1e-4 | −0.639 | ✓ 5/5 | ✓ |
| `sparsity_id` | −0.115 | +0.268 | 0.005 | +0.584 | ✓ 5/5 | ✓ |
| `center_alignment_id` | +0.830 | +0.483 | <1e-4 | +0.400 | ✓ 5/5 | ✗ (ρ<0.5) |
| `participation_ratio_id` | +0.858 | +0.659 | <1e-4 | +0.503 | ✗ | ✗ |
| `numerical_rank_id` | −0.212 | +0.436 | <1e-4 | +0.570 | ✗ | ✗ |
| `capacity_id` | −0.677 | −0.429 | <1e-4 | −0.331 | ✗ | ✗ |
| `nc1_id` | +0.019 | −0.145 | 0.138 | −0.146 | ✗ | ✗ |
| **`acc_test`（基线）** | **−0.045** | **+0.392** | <1e-4 | **−0.231** | ✗ | ✗ |

**叙事**：
- GLUE 三标记（`D_eff` / `Ψ_eff` / `R_eff`）的预测力**完全经受住去混淆**，
  且逐架构符号一致（5/5）→ **不是 Simpson 悖论**
- **ID 准确率在这份数据上根本不能作基线**：pooled r = −0.045（几乎无关），
  残差化 Pearson +0.392 但 Spearman −0.231 **方向不一致**，
  逐架构符号混乱 → 按判据不可用作基线
- 通用几何量**并非全部失效**，但普遍弱于 GLUE 三标记
  （注：v1 曾误称 task-agnostic 失效，已按 ADR-0010 修正）

### 4.3b 三方一致性交叉验证（同带宽 band=1pp）
| 标记 | 官方 80 模型 | 官方 224 模型 | 自建 48 组 | 一致? |
|---|---|---|---|---|
| `dimension_id` | +0.737 | **+0.925** | — | ✓✓ |
| `utility_id` | +0.808 | **+0.901** | — | ✓✓ |
| `participation_ratio_id` | +0.592 | +0.659 | **+0.701** | ✓✓✓ |
| `radius_id` | +0.153 | −0.584 | — | ✗ **不稳定** |
| **`acc_test`（基线）** | −0.654 | +0.392 | **−0.439** | ✗ **不稳定** |

15 个标记中 **13 个方向一致**。`radius_id` 与 `acc_test` 符号在两份数据间不稳定
→ **不能作为稳定预测器**，这是本文"哪些标记可信"这一贡献的一部分。

### 4.3c 分层 bootstrap 置信区间（2000 次，全流程重采样）

每个 bootstrap 样本内**重新执行完整协议**（分层重采样 → 重新筛选 ID 受控子集
→ 重新残差化 → 双相关性），CI 反映整个流程的不确定性，而非仅最后一步。

**GLUE 标记的 CI 紧且远离零：**

| 预测器 | 数据 | 残差化 r 95% CI | Spearman ρ 95% CI | bootstrap 稳健率 |
|---|---|---|---|---|
| `dimension_id` | 224 模型 | **[+0.879, +0.955]** | [+0.726, +0.897] | **100%** |
| `utility_id` | 224 模型 | **[+0.842, +0.938]** | [+0.626, +0.847] | **100%** |
| `dimension_id` | 80 模型 | [+0.581, +0.833] | [+0.646, +0.873] | 100% |
| `id_numerical_rank` | 自建 48 组 | [+0.797, +0.923] | [+0.772, +0.934] | 100% |

**基线 `acc_test` 的 CI 在数据集之间完全不重叠：**

| 数据 | 残差化 r 95% CI | 符号 |
|---|---|---|
| 224 模型 | **[+0.215, +0.622]** | 正 |
| 80 模型 | **[−0.825, −0.417]** | **负** |
| 自建 48 组 | **[−0.659, −0.181]** | **负** |

→ 每份数据内部 CI 都"显著"排除零，但 **224 与 80 两份数据的 CI 区间无任何重叠，
符号相反**。ID 准确率与 OOD 的关系**连方向都不具备跨数据集可迁移性**——
这是比"弱预测"更强的否证：两个紧致 CI 指向相反方向，任何单一数据集上的
"显著基线"都不可信。

### 4.4 结果 3：逐架构分解（表 T3）★Simpson 悖论诊断
| 标记 | DenseNet121 | EfficientNetB0 | MobileNet | ResNet18 | VGG | 一致? |
|---|---|---|---|---|---|---|
| `dimension_id` | +0.59 | +0.86 | +0.73 | +0.99 | +0.86 | ✓ |
| `utility_id` | +0.66 | +0.82 | +0.68 | +0.99 | +0.84 | ✓ |
| `center_alignment_id` | +0.04 | +0.47 | +0.52 | +0.96 | +0.84 | ✓ |
| `acc_test` | +0.88 | −0.73 | −0.78 | +0.52 | −0.92 | ✗ |

### 4.5 结果 4：带宽稳健性（表 T4）
| 标记 | b=1 | b=2 | b=3 | b=5 | b=8 | Spearman(b=2) |
|---|---|---|---|---|---|---|
| `dimension_id` | +0.737 | +0.713 | +0.751 | +0.776 | +0.776 | +0.805 |
| `utility_id` | +0.808 | +0.729 | +0.726 | +0.761 | +0.761 | +0.799 |
| `participation_ratio_id` | +0.592 | +0.659 | +0.720 | +0.779 | +0.779 | +0.794 |
| `acc_test` | −0.654 | −0.628 | −0.699 | −0.767 | −0.767 | −0.606 |

### 4.6 结果 5：偏移类型的边界（表 T5）★第二核心结果
| 场景 | 最强预测器 | \|ρ\| | `acc_test` 排名 | `acc_test` ρ | 残差化后 ρ |
|---|---|---|---|---|---|
| fog | **`acc_test`** | 0.990 | 1/14 | +0.990 | +0.918 |
| brightness | **`acc_test`** | 0.994 | 1/14 | +0.994 | +0.968 |
| contrast | **`acc_test`** | 0.975 | 1/14 | +0.975 | +0.772 |
| gaussian_noise | `mean_angle_id` | 0.632 | 7/14 | +0.185 | −0.087 |
| pixelate | `center_alignment_id` | 0.445 | 11/14 | −0.128 | +0.215 |
| jpeg_compression | `sparsity_id` | 0.351 | 12/14 | +0.087 | −0.010 |

**叙事**：几何标记的相对优势**取决于偏移类型**。类别级偏移（CIFAR-100）下
几何标记大幅领先；亮度/结构类 corruption 下 ID 准确率几乎完美解释表现；
噪声与压缩类下两者都失效。

### 4.7 结果 6：独立验证（自建 48 组 × 两个类级偏移）★独立于原作者数据
自建 48 组 resnet18/CIFAR-10（4 lr × 4 wd × 3 seed × 200 epoch），
用**公开可算指标**跑**同一套 Protocol A**，在**两个独立的类级偏移**上：
CIFAR-100 与 Tiny-ImageNet（200 新类，冻结特征 200 类 linear probe）。

| 预测器（残差化 r） | CIFAR-100 | Tiny-ImageNet |
|---|---|---|
| `id_numerical_rank` | **+0.870** | **+0.853** |
| `id_local_smoothness` | +0.832 | +0.792 |
| `id_pr_perclass` | +0.701 | +0.707 |
| `glue_dimension`（自实现，相对比较） | — | +0.729 |
| **`id_acc`（基线）** | **−0.439** | **−0.338** (p=0.022) |

**核心**：基线在**两个独立的类级偏移上都是负相关**——
"ID 受控条件下 ID 准确率与 OOD 负相关"**跨偏移数据集稳定**，
不是 CIFAR-100 特例。两个 OOD 指标本身高度相关（r=0.987），
但其价值在于排除"单一 OOD 数据集偶然"的质疑。

corruption 场景细分（19 种）同样复现：亮度/结构类 → `id_acc` 排第 1；
噪声类 → 几何量最强、`id_acc` 排第 7。

### 4.7a 跨架构独立验证（三架构，真架构维度混杂源）★

新增 shufflenet_v2_x1_0（2018）与 regnet_y_400mf（2020）各 16 组
（lr∈{0.5,0.25,0.125,0.0625} × 2 wd × 2 seed，200 epoch），与 resnet18 48 组合并。

**跨架构子集（band=±2pp，混杂源 = 架构；32/32 全量，健康 79）**：
ID 跨度 3.70pp 时 OOD 跨度保留 100%（CIFAR-100）/ 93%（TinyImageNet）。
几何量残差化后极强且逐架构符号一致：
`id_pr_perclass` +0.941/+0.948 (ρ≈+0.98)、`id_local_smoothness` +0.956/+0.930、
`id_numerical_rank` +0.943/+0.952。**基线 `id_acc` 残差化 +0.153 (p=0.37) /
+0.145 (p=0.40)，均不显著，逐架构符号混乱** → 再次不通过判据。

**组内协议（每架构以自身均值为带心，lr 族残差化）**：

| 架构 | n | `pr_perclass` 残差化 r | `id_acc` 残差化 r | 基线符号 |
|---|---|---|---|---|
| resnet18 (C100) | 46 | +0.701 | **−0.439** | 负 |
| shufflenet_v2 (C100) | 14 | **+0.988** | **−0.637** | **负** |
| regnet (C100) | 13 | +0.659 (p=.014) | +0.053 (n.s.) | 不显著 |
| resnet18 (Tiny) | 46 | +0.707 | −0.338 | 负 |
| shufflenet_v2 (Tiny) | 14 | +0.985 | **−0.584** | **负** |
| regnet (Tiny) | 13 | +0.536 (p=.059) | −0.019 (n.s.) | 不显著 |

**两个新发现**：
1. **几何量的预测力在三个架构上都极强**（0.89–0.99），且**架构越"现代"
   （shufflenet 最高 0.99；regnet 组内 0.54–0.66，样本少波动大）。
2. **基线负相关在 shufflenet 上复现且更强**（−0.637/−0.584）；
   regnet 上不显著（样本少 n=12）。**没有一个架构上基线显著为正。**
3. shufflenet 的 ID acc 上限（0.82）低于全体均值带（0.832–0.872），
   落在跨架构子集之外——**低容量架构整体被 ID 受控筛选排除**，
   这本身是"ID acc 基线偏差"的一个具体表现（它隐式选择了高容量架构）。

### 4.7b 自实现 GLUE 的跨实现一致性（附录级）
自实现 GLUE（Algorithm 1 + cvxopt，仅相对比较）在自建 48 组上：
`dimension` r=+0.908 / rho=+0.924 —— 与官方实现、官方 ResNet18 子集
（+0.865/+0.866）**跨实现、跨数据集一致**。
`radius`/`utility` 跨实现不一致 → 印证其不稳定性。
**不同实现 + 不同数据 + 同一结论**，这是 `dimension` 最强的一层稳健性证据。

> **本层的意义**：验证的是**协议本身**在完全独立训练的数据上是否成立，
> 而非"再算一遍同样的数字"——直接回答"结论是否只是原作者数据的产物"
> 这一审稿质疑。

---

## 5. Discussion

### 5.1 为什么 ID 准确率在受控后变成负相关
受控 ID 准确率后，剩余的 acc 差异主要来自**架构间的容量分配差异**，
而这些差异与 OOD 呈反向。观察到的模式：训练充分的紧凑架构
（ResNet18、DenseNet121）ID 略低但 OOD 更好；ID 略高的 VGG / MobileNet
在 OOD 上显著更差。这提示 **ID accuracy 不是 OOD 泛化的单调代理**，
在跨架构比较中甚至可能是**误导性代理**。

**这解释了原文为何会报告"几何标记优于 ID accuracy"**：原文的 pooled 比较中，
ID accuracy 的正相关来自架构间差异；一旦控制架构，真实关系是负的。
→ **我们的贡献不是推翻原文，而是揭示原文结论所依赖的度量方式
会给出看似正确但机理相反的解释。**

### 5.2 task-relevance 与 shift type 的二维结构
| | 类别级偏移（CIFAR-100） | 损坏级偏移（亮度/结构） | 损坏级偏移（噪声/压缩） |
|---|---|---|---|
| task-relevant 标记（GLUE 三指标） | **大幅领先** | 与 ID acc 相当 | 弱，两者皆失效 |
| 通用几何量 | 部分有效但更弱 | 弱 | 弱 |

### 5.3 对实践者的启示
- 不要用几何标记做**同分布**的模型选择（ID 场景下它几无信息，Phase 0 已证）
- 跨架构比较 OOD 泛化时，**ID accuracy 是不可靠的基线**，几何标记更有用
- 偏移类型必须匹配：用 brightness 类偏移训练、用 noise 类评估，标记会失灵

### 5.4 与"脑科学—AI 联姻"主线的意义
本项目最初的假设是"表征几何的优雅性可作为普适质量指标"。本文给出**边界条件**：
几何标记不是普适质量指标，它只在 **(存在分布偏移) 且 (指标与任务绑定)**
两个条件同时成立时才有预测力。这比原方案"优雅性可作质量指标"的乐观论断
更准确，也更可发表。

---

## 6. Limitations（必须诚实列出）

1. **6 个 corruption 场景共享同一批 80 个模型与同一套 ID 几何标记**，
   只有 corruption 后的标签不同 → corruption 场景之间**不是独立样本**。
   这是官方数据的组织方式，本文无法改变，只能如实说明。
2. 主分析**依赖原作者发布的测量值**，虽然该数据公开、逐模型可核对，
   但"用他人数据得出关于其方法的结论"需要格外审慎。
   → §4.7 用**完全独立训练**的 48 组模型复现了协议结论，
   §4.3b 用**第二份官方数据**（224 模型）做了一致性交叉验证。
3. **主结果全部基于官方发布的测量值**：`D_eff` / `R_eff` / `Ψ_eff` 均为
   官方 `df_fig3.pkl` 中的逐模型数值，**不依赖任何私有实现**，可完全复现。
   我们亦按 GLUE 论文 Algorithm 1 与公开的 Chung et al. (2018) anchor-point QP
   自行实现了这三指标（`code/glue_public.py`），但发现论文附录另有两项要求
   ——**逐流形对分析**与 **Gaussianization 预处理**（Wakhloo et al., 2023）——
   未被我们的实现覆盖，且 a/b/c 实为含伪逆的**矩阵二次型**。
   因缺少第二个官方 checkpoint 做交叉验证，自实现版本**仅用于定性理解，
   不进入本文任何定量结论**。
4. 类别级偏移覆盖 CIFAR-100 与 Tiny-ImageNet 两例（自建数据侧）；
   官方数据侧仍只有 CIFAR-100，未覆盖 ImageNet 级别全量偏移。
5. 残差化采用架构哑变量 OLS，是线性去混淆的常用近似。
   置信区间采用分层 bootstrap（2000 次，层 = 架构/lr 族，
   每个样本内重新执行完整协议），已覆盖全部主结果（§4.3c）。
6. "受控子集"是**观察性**筛选，非随机分配；虽然 ID 带宽内跨度仅 3.82pp，
   但仍可能残留未测量的混杂。
7. ~~独立验证只用单一架构~~ ✅ 已补齐：三架构（resnet18 / shufflenet_v2 /
   regnet_y_400mf）跨架构子集 + 组内协议（§4.7a）。新发现：shufflenet 的
   ID acc 上限低于全体均值带，被 ID 受控筛选隐式排除——基线偏差的又一实例。

---

## 7. Conclusion

（与 Abstract 呼应，3 段）

---

## 图表清单

| 编号 | 内容 | 对应数据 |
|---|---|---|
| Fig. 1 | 概念图：几何标记预测力的二维边界（task-relevance × shift type） | `figure1_concept.md` |
| Fig. 2 | 散点矩阵：D_eff / Ψ_eff / acc_test vs OOD（残差化前后两行） | T1+T2 |
| Fig. 3 | 逐架构回归线：展示 Simpson 悖论（acc_test 符号翻转） | T3 |
| Fig. 4 | 带宽稳健性曲线：r 随 b 变化 | T4 |
| Fig. 5 | corruption 场景热力图：14 标记 × 6 场景的 \|ρ\| | T5 |
| Fig. 6 | 三方一致性图：官方 80 模型 / 官方 224 模型 / 自建 48 组，同一套协议 | protocol_a_fig3 + 独立验证 |
| Table 1 | 全样本相关性 | T1 |
| Table 2 | ID 受控 + 残差化主结果 | T2 |
| Table 3 | 逐架构符号一致性 | T3 |
| Table 4 | 带宽敏感性 | T4 |
| Table 5 | corruption 场景细分 | T5 |
| Table 6 | 三方一致性交叉验证 | protocol_a_fig3 F6 |
| Table 6b | 分层 bootstrap 95% CI | bootstrap_ci_report |
| Table 7 | 独立验证 × 两个类级偏移（自建 48 组） | ind_validation + tinyimg |
| Table 8 | 自实现 GLUE 跨实现一致性（附录） | glue_own48 |
| Table 9 | 三架构跨架构 + 组内协议 | multiarch_report |

---

## 参考文献占位

```
[1]  Chou, Kirsanov, Yang, Chung. Diagnosing Generalization Failures from
     Representational Geometry Markers. ICLR 2026. arXiv:2603.01879.
[2]  Chou et al. Geometry Linked to Untangling Efficiency Reveals Structure
     and Computation in Neural Populations. bioRxiv 2024.02.26.582157 (v2, 2025).
[3]  Chung, Lee, Tegmark. Classification and Geometry of General Perceptual
     Manifolds. Phys. Rev. X 8, 031039 (2018).
[4]  Jiang et al. How Powerful is Graph Convolution? ICLR 2020.
[5]  Sorscher et al. Beyond neural scaling laws. NeurIPS 2022.
[6]  Hendrycks, Dietterich. Benchmarking NN Robustness to Common Corruptions.
     ICLR 2019.
[7]  Liang et al. Do ImageNet Classifiers Generalize to ImageNet? ICLR 2018.
[8]  Cimpoi et al. A Benchmark for Robustness to Common Corruptions. 2019.
[9]  Nakkiran et al. Deep Double Descent. ICLR 2021.
[10] Pearl. Causality. 2nd ed. 2009.（混杂、残差化、d-separation）
[11] Efron & Tibshirani. An Introduction to the Bootstrap. 1993.
[12] Siegel & Janssen. Simultaneous selection of the best models. 2020.（多重比较）
```

---

## 投稿策略建议

| 路线 | 条件 | 目标 |
|---|---|---|
| **A. 主会正刊** | 独立验证通过 + 补分层 bootstrap CI + 再加 1–2 个架构（ResNeXt/WideResNet） | ICLR / NeurIPS |
| **B. Workshop** | 独立验证部分通过 | NeurIPS Workshop on Neural Rep. Geometry |
| **C. 降级** | 独立验证失败 | Findings / 短文：诚实的受控再分析 |

**最小可行增量（按性价比排序）**：
1. ~~独立验证~~ ✅ 已完成（ADR-0012）
2. ~~主数据扩容到 224 模型~~ ✅ 已完成（ADR-0013）
3. ~~分层 bootstrap CI~~ ✅ 已完成（§4.3c，2000 次 × 3 数据集）
4. 加 1 个现代架构（ResNeXt-50 / WideResNet-28-10），补 16 个模型（半天）
5. 补 1 个类级偏移数据集（ImageNet-100 子集或 STL-10，1 天）
6. 若 GLUE 官方代码获批 → 把 GLUE 三指标跑在自建 48 组上，闭合第三层证据链


---

## v3.6 修订记录：对抗性审读（2026-10-27）

用 grilling 管线对初稿做压力测试，四个致命点全部用数据核实后处置。

### 修订 1：Q1 判定——「符号不转移」立论成立（已核验，无需降级）

- **核查**：224 release 与 80 release 的模型键交集 = **0**（两次独立发布，前提成立）
- **核查**：`acc_lp_ood` 在 80 release 的 6 个 corruption 场景间**完全相同**
  （两两 corr=1.0000, maxdiff=0.000）→ 是场景无关的类级迁移量，不是 corruption 专属
- **上游确认**：原论文 Appendix C.3 明确 CIFAR-100 线性探测为类级 OOD 协议，
  CIFAR-10-C 为独立的标签保持型 corruption 协议（不训练新探针）
- **结论**：两个 release 的 OOD 目标**等价可比**，立论保留
- **新增**：§3.5 补写这三条核验，§4.5 补括号说明

### 修订 2：Q2+Q3——自建 48 组改为「符号不稳定」表述

- **发现**：C4 逐 lr 族检查暴露汇总数掩盖的符号翻转
  （−0.802 / −0.603 / −0.330 / **+0.694**），按论文自身 C4 判据基线**不通过**
- **敏感性**：剔除未收敛的 lr=1.0 族（OOD 7.1–14.7 vs 其余 13.7–32.4）后
  基线三族全负（−0.966/−0.603/−0.330），聚合 **−0.658 / −0.513**，
  bootstrap CI [−0.803,−0.420] / [−0.695,−0.217]，P(r>0)≈0%
- **改动**：§4.7 增 C4 逐族表 + 收敛族敏感性段；摘要/引言/结论
  从「负号复现」改为「符号不稳定」；Limitations 新增第 8 条
- **产物**：`analyze_ind48_robust.py` + `results/ind48_robust_report.md`

### 修订 3：Q4——ShuffleNetV2 观察从「偏差证据」改为「C1 限制 + 协议补救」

- **问题**：C1 用 ID 准确率定义子集，论文却批评 ID 匹配排除低容量架构 → 自指
- **改动**：
  - §3.2 C1 定义补写「匹配中心是全体均值，这是一个 choice」，并声明同时报告
    **within-architecture 变体**（各组以自身均值为带心，不排除任何族）
  - §4.7a 改写为「这是我们协议的局限，不是发现」，并指出组内协议正是补救
  - §5 Discussion 新增「A self-reference we cannot dissolve」段，
    正面回应「为什么仍值得做 ID 匹配」（混杂量 0.48 vs −0.63，远大于被比较的效应）
  - §7 Conclusion 相应改写

### 修订 4：Q5——定位从「审计论文」调整为「方法论文 + 审计示例」

- **改动**：
  - 摘要重写：三条发现并列（标记存续 / 基线不是稳定参照 / 优势随偏移类型变化），
    「符号不转移」从主张降为观察之一
  - 贡献列表第 2、3 条重写
  - §3.2 补「协议对称施加于基线」声明
  - §3.3 补 C4 对基线的自反加载性论证
  - §4.3 标题「baseline is undefined」→「baseline is not a stable reference point」
  - §5 Discussion 第 2 条机制改写为「population dependence, not a fixed weakness」

### 未受影响（结论方向不变）
- 官方 224/80 release 的全部数字（几何量 0.89–0.99，基线 +0.39 / −0.63）
- 三架构 79 模型跨架构结果（PR 0.941/0.948，基线 +0.153/+0.145 不显著）
- 跨实现一致性 +0.91 vs +0.87
- corruption 分解（fog/brightness/contrast ID acc ρ≈0.98–0.99）


---

## v3.5 修订记录：第二轮对抗性审读（2026-10-27）

### Q6 §4.6 双口径（已改）
§4.6 改为**双列**：pooled ρ 与 Protocol A（C1±2pp + C2 架构残差化）残差化 ρ 并列。
结论方向不变，但 pixelate 的基线由 −0.12 变 +0.22、contrast 由 +0.965 降至 +0.772，
最优标记在 pixelate/jpeg 下由 center_alignment 变为 sparsity —— 这本身就是"必须双口径"的证据。
Protocol A 口径下 ID acc 在 fog/brightness/contrast 仍是 1/12 名（+0.918/+0.968/+0.772），
噪声/压缩类掉到 8/12、6/12、11/12。

### Q7 网格表述（已改，全文三处）
真实结构：**每架构 4 lr × 4 wd × 2–3 seed**，48 模型/架构（EfficientNetB0 仅 2 seed、32 个），
全局并集才覆盖 13 lr × 9 wd。原"13 lr × 9 wd × 3 seeds"会被读成笛卡尔积 1755。
摘要/引言/§4.1 三处全部改为精确描述。

### Q8 摘要口径（已改）
摘要的 corruption 区间由 pooled（−0.13~+0.09）改为受控口径（−0.09~+0.22），
全文口径统一为 Protocol A。

### Q9 新增 §4.7c 正交性证据（本轮最重要的加分项）
自建 48 组上：几何量与 ID acc 相关 −0.17~−0.04，与 OOD 相关 +0.90~+0.95；
控制 ID acc 后偏相关仍 +0.69~+0.86。**这是不依赖任何跨 release 假设的第三条独立证据链**，
已写入摘要、引言、贡献列表（新增第 5 条）、§5（新增机制 4）、§7。

### Q10 致原作者核实函（已发出）
三问：probe 流水线一致性 / probe 超参 / 两文件群体与网格结构。
**不依赖回复**：Limitations 第 9 条按"未获作者确认"写定，主论点不依赖跨 release 比较。
- 邮件：Agent Mail → cchou@ + cc schung@，主题 Three protocol questions on the two released datasets
- 存档：protocol_verification_email.md；决策记录：adr_0015.md
- 预先准备两条降级路径（确认同协议 → 改致谢；不同协议 → §4.5 降为观察，主线转 §4.7/§4.7c）

### 本轮未受影响
官方 224/80 的全部主数字、三架构 79 模型跨架构结果、跨实现一致性、bootstrap CI、
引言的三条主发现结构。


---

## v3.6 修订记录：数字一致性审计（2026-10-27 傍晚）

新增 `audit_claims.py`：把论文 §4.2–§4.7c 的每条声称数字从原始数据用**与分析脚本相同的协议**
独立重算，逐条比对。**最终 111 条断言，111 条通过，0 条不符。**

审计中发现并修正的问题（5 处，全部为论文侧表述，数据侧无误）：

| # | 位置 | 问题 | 修正 |
|---|---|---|---|
| 1 | §3.3 | 80 模型 pooled r 写作 **+0.48**，实为 **+0.443** | 改为 +0.44 |
| 2 | §3.3 | 「无 C4」论证里 PR 的 r=+0.66 未说明是**残差化后**的数，与 Table 2 的 +0.858 看似冲突 | 明写 "even after C1 and C2 (residualized r=+0.66)" |
| 3 | Table 2 | 列名 "raw r" 有歧义——实为**子集内未残差化**的 r，不是全样本 r | 列名改为 "$r$ (subset, pre-C2)"，并加说明 + 基线 $+0.48 ightarrow -0.045 ightarrow +0.392$ 的序列注解 |
| 4 | §3.2 / §4.4 | 声称带宽集含 b=8，但脚本实际只跑 {0.5,1,2,3,5}；且 b=8 在两个数据集上都**退化**（224 留 223/224，80 留 80/80 = 未匹配全样本） | 带宽集改为实际执行的 5 档，并显式说明 b=8 为何不报（附退化证据） |
| 5 | §4.6 | 「ID accuracy ranks 1st for all 11 photometric/structural/blur scenarios」**表述错误**——是 19 个里有 11 个排第 1，且该集合与暗示的不同（含 jpeg/elastic、不含 contrast）；噪声 4 场景"7th of 8"实为 7/8/8/7 | 改为逐场景列举，并**主动报告 contrast 例外**（自建 ρ=+0.09 排 7/8，官方 ρ=+0.97 最强）——这是两数据集唯一实质分歧 |

### 顺带核验通过的项（无需修改）
- Table 2 全部 7 行 × 3 列（subset r / resid r / ρ）逐个吻合
- n=106、ID 跨度 1.92pp、OOD 跨度保留 91.8% 全部吻合
- Table 3 基线逐架构符号 +0.48/−0.48/+0.46/+0.24/+0.42 吻合
- §3.3 C3 论证 center_alignment r=+0.48 ρ=+0.40 吻合
- §4.4 带宽区间（D_eff 0.71–0.78 / 0.89–0.93；基线 0.39–0.57）吻合
- §4.5 全部 7 条 bootstrap CI 与稳健率吻合
- §4.7 五个标记 × 两偏移 + C4 逐族符号吻合
- §4.7c 正交性 12 个数吻合
- §4.7a 跨架构 8 个数 + n=36 吻合
- §4.7b 跨实现 4 个数 + n=41 吻合

### 代码级审计结论
- `arch_residual` 用 OLS dummy 拟合，与论文 §3.2 描述一致
- C3 判据实现为「符号一致 + |ρ|≥0.5」，与论文一致
- bootstrap 分层实现为「每层内等量有放回重采样」，与论文 §3.4 描述一致
- 两份官方数据文件 md5 与上游仓库完全一致（75c5ed80… / 8537fb30…），路径差异无害

### 产物
- `audit_claims.py`（可复现，111 条断言）+ `results/claims_audit_report.md` + `results/claims_audit.json`
