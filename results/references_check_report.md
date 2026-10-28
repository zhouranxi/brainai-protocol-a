# 引用核查报告 — paper_draft_v1.md

核查日期：2026-10-27 · 方法：cite-hallucination-check 管线（正文内嵌引用逐条联网核验原始出处）

## 红旗清单（按严重度）

### ⚠ 结构性发现（最重要）

**初稿原本没有 References 节**——全部引用均为正文内嵌 author-year，无文献表。
已处理：本次核查后补全 References 节（12 条），并更正 1 处年份错误。

### exist-minor（1 条，已修正）

| 引用原文 | 判定 | 对不上的组件 | 真实来源 | 动作 |
|---|---|---|---|---|
| `(Nakkiran et al., 2021)` double descent | **exist-minor** | 日期 not match | Deep Double Descent: Where Bigger Models and More Data Hurt — Nakkiran, Kaplun, Bansal, Yang, Barak, Sutskever, **ICLR 2020**（arXiv 1912.02292, 2019.12） | 正文年份已改 2020，References 已列 |

### 语义提示（1 条，建议核对措辞，非错误）

| 引用 | 判定 | 说明 |
|---|---|---|
| `(Sorscher et al., 2022)` "regression-oriented accounts link generalization to the parsimony of internal representations" | exist，措辞需核对 | 来源真实（PNAS 119(43) e2200800119）。但该文核心主张是**高维流形有利于** few-shot 学习——与"parsimony（简约）"的单向表述有张力。建议把该句措辞对齐原文（几何性质决定可学性，而非"越简约越好"）。 |

### 缺引（3 处，已补入 References）

| 正文提及 | 应补引用 |
|---|---|
| RegNetY-400MF (2020) | Radosavovic et al., Designing Network Design Spaces, CVPR 2020 |
| ShuffleNetV2 (2018) | Ma et al., ShuffleNet V2, ECCV 2018 |
| Tiny-ImageNet（数据集） | Le & Yang, 2015（CS231n 技术报告） |

## 逐条核验结果（9/9 真实可溯）

| # | 引用 | 判定 | 真实出处（核验源） |
|---|---|---|---|
| 1 | Chou, Kirsanov, Yang & Chung, ICLR 2026（"the original study"） | **exist** | Diagnosing Generalization Failures from Representational Geometry Markers，作者四人逐一吻合，arXiv:2603.01879，Comments 明确 "Published in ICLR 2026" |
| 2 | Chou et al., 2025（GLUE 框架） | **exist** | Chou, Le, Wang, Chung — Feature Learning beyond the Lazy-Rich Dichotomy, ICML 2025，arXiv:2503.18114；GLUE（Geometry Linked to Untangling Efficiency）及 D_eff/R_eff/Ψ_eff 锚点 QP 定义均在此文 |
| 3 | Chung, Lee & Sompolinsky, 2018（流形容量理论） | **exist** | Classification and Geometry of General Perceptual Manifolds, PRX 8, 031003；GLUE 两篇论文均引其为容量理论源头 |
| 4 | Chung et al., 2018（anchor-point solver） | **exist** | 同上（同一篇；锚点 QP 即该文定义） |
| 5 | Sorscher et al., 2022 | **exist**（措辞见上） | PNAS 119(43), e2200800119, 2022-10 |
| 6 | Pearl, 2009 | **exist** | Causality (2nd ed.), Cambridge Univ. Press（公认经典） |
| 7 | Nakkiran et al., 2021 → 2020 | **exist-minor** | ICLR 2020（dblp + Scholar 双源确认） |
| 8 | Jiang et al., 2020（有效秩类泛化指标） | **exist** | Fantastic Generalization Measures and Where to Find Them, ICLR 2020；40+ 复杂度度量大规模综述，作 umbrella 引用恰当 |
| 9 | Hendrycks & Dietterich, 2019 | **exist** | Benchmarking NN Robustness to Common Corruptions and Perturbations, ICLR 2019（CIFAR-10C 出处） |
| 10 | Efron & Tibshirani, 1993 | **exist** | An Introduction to the Bootstrap（公认经典） |

## 小提示

- Chou 2025 在后续文献中被引为 **2025a / 2025b 两篇**（GLUE 理论 vs feature-learning 分析）。
  我们引用的 ICML 2025 论文（arXiv:2503.18114）两者内容都含；若最终投稿需与原研究编号体系
  对齐，届时确认用 a 还是 b。
- 本地 `glue_paper.pdf` / `ilm_paper.pdf` 即被引论文原文，排版时可再逐句对照。

## 统计

**12 条引用/提及 → 真实可溯 9、小错 1（已修正）、编造 0、缺引 3（已补）。编造率 0%。**
