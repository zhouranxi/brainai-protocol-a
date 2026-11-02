# 投稿前联合核查报告 — 引用真实性 / 论文-代码-数据一致性

核查日期：2026-11-02 · 范围：`paper/main.tex`（R1，19 页）与 `paper/measurement/main.tex`（测量论文，9 页）
方法：`cite-hallucination-check` 管线（联网核验原始出处）+ 独立重算审计（不依赖既有审计脚本的报告）

---

## 摘要

| 维度 | 结论 |
|---|---|
| **引用真实性** | **12 + 4 条全部真实可溯，0 编造**。8 个 arXiv 编号经 arXiv 官方 API 逐一验证存在且与条目一一对应；4 条非 arXiv 引用经 CrossRef 验证。**唯一缺陷是格式**：R1 的 `references.bib` 有 6 个 `note` 字段落在条目之外被 BibTeX 静默丢弃 → PDF 里那 6 条引用缺 arXiv 编号。 |
| **论文-代码-数据一致性** | 306 处定量断言全部重算核对，**297 一致 / 9 不一致**。其中 **1 处阻断投稿**（统计量类型错标导致结论身份改变）、**4 处需修改**、**4 处建议**。 |
| **审计盲区** | **35 处**论文写了但 CI 审计脚本未覆盖的定量断言；其中 1 处审计检查是**恒真的假保护**。 |
| **模型数口径** | 论文写 78，实际 trained=80 / healthy=79（1 个发散），三处自相矛盾。 |

---

## 第一部分：引用真实性核查

### 1.1 红旗清单

| 严重度 | 条目 | 判定 | 说明 |
|---|---|---|---|
| **需修改（格式）** | `paper/references.bib` | `exist-minor` | 6 处注释被截断，`note` 与闭合 `}` 落到下一条目之前。**编号归属全部正确**，但 BibTeX 会丢弃这些孤立 `note`，导致 6 条引用在 PDF 中缺失 arXiv 编号 |
| **需修改（格式）** | `chou2026diagnosing` / `chou2025feature` 的 `booktitle` | `exist-minor` | ICLR/ICML 官方 BibTeX 为 "The Fourteenth International Conference on Learning Representations" / "Forty-second International Conference on Machine Learning"，本文写法与官方不一致 |
| ✅ 无问题 | 其余全部 | `exist` | 见下方逐条核验 |

### 1.2 逐条核验结果（16/16 真实可溯）

**arXiv 编号 — 经 `export.arxiv.org/api/query` 逐一验证**

| arXiv ID | 真实标题 | 真实作者 | 论文中的条目 | 判定 |
|---|---|---|---|---|
| 2603.01879 | Diagnosing Generalization Failures from Representational Geometry Markers | Chou, Kirsanov, Yang, Chung | `chou2026diagnosing` | ✅ exist（ICLR 2026 已确认，proceedings.iclr.cc 有正式版） |
| 2503.18114 | Feature Learning beyond the Lazy-Rich Dichotomy | Chou, Le, Wang, Chung | `chou2025feature` | ✅ exist（ICML 2025 spotlight） |
| 1710.06487 | Classification and Geometry of General Perceptual Manifolds | Chung, Lee, Sompolinsky | `chung2018classification` | ✅ exist（PRX 8, 031003, 2018） |
| 1903.12261 | Benchmarking NN Robustness to Common Corruptions | Hendrycks, Dietterich | `hendrycks2019benchmarking` | ✅ exist（ICLR 2019） |
| 1912.02178 | Fantastic Generalization Measures and Where to Find Them | Jiang, Neyshabur, Mobahi, Krishnan, Bengio | `jiang2020fantastic` | ✅ exist（ICLR 2020） |
| 1807.11164 | ShuffleNet V2 | Ma, Zhang, Zheng, Sun | `ma2018shufflenet` | ✅ exist（ECCV 2018, LNCS） |
| 1912.02292 | Deep Double Descent | Nakkiran, Kaplun, Bansal, Yang, Barak, Sutskever | `nakkiran2020deep` | ✅ exist（ICLR 2020） |
| 2003.13678 | Designing Network Design Spaces | Radosavovic, Kosaraju, Girshick, He, Dollár | `radosavovic2020designing` | ✅ exist（CVPR 2020） |
| — | An Introduction to the Bootstrap | Efron, Tibshirani | `efron1993bootstrap` | ✅ exist（CrossRef DOI 10.1007/978-1-4899-4541-9, 1993） |
| — | Causality: Models, Reasoning, and Inference (2nd ed.) | Pearl | `pearl2009causality` | ✅ exist（Cambridge UP 2009；书籍不在 arXiv/CrossRef，符合预期） |
| — | Tiny ImageNet Visual Recognition Challenge | Le, Yang | `le2015tiny` | ✅ exist（CS231n 技术报告 2015） |

**measurement 论文 4 条**：`chou2026diagnosing`、`chung2018classification`、`jiang2020fantastic` 均与上表同一真实来源；`zhou2026protocol` 为自引（Under review，标记正确）。

### 1.3 bib 结构缺陷的具体形态

`paper/references.bib` 有 12 个条目，但只有 2 个带 `note` 字段——**另外 6 个 note 字段被夹在条目之间**（BibTeX 静默丢弃），另有 12 个悬空的行首 `}`：

```
36  % Stratified bootstrap,          ← 注释被截断
37    note      = {arXiv:1710.06487}   ← 落在 chung2018classification 条目之外
38  }
39  @book{efron1993bootstrap,         ← 这个 } 是上一条残留，efron 条目因此少一个 note
```

**关键判断**：编号与条目的对应关系**全部正确**（注释文字与编号错位，但编号本身没挂错条目）。因此这**不构成编造**，只是 6 条引用在 PDF 里少了 arXiv 编号。

> 注：`paper/measurement/references.bib` 的 4 个 note 字段**全部位置正确**，且编号全部正确。

### 1.4 建议动作

1. 修复 `paper/references.bib` 的 6 处截断（把 note 移回对应条目），并修正 6 条注释文字与编号的错位
2. 两处 `booktitle` 改为 ICLR/ICML 官方写法
3. 让 `paper/measurement/references.bib` 与 R1 保持一致（同一来源的 note 应一致）

---

## 第二部分：论文-代码-数据一致性

### 一致性总览

| 论文章节 | 断言数 | 已核对 | 一致 | 不一致 | 盲区 |
|---|---|---|---|---|---|
| R1 摘要 | 18 | 18 | 18 | 0 | 3 |
| R1 §4.2 全样本景观 | 4 | 4 | 4 | 0 | 0 |
| R1 表 2 主结果 | 31 | 31 | 31 | 0 | 0 |
| R1 §3.3 条件承重论证 | 6 | 6 | 6 | 0 | 4 |
| R1 §4.4 带宽敏感性 | 8 | 8 | 8 | 0 | 1 |
| R1 表 3 Bootstrap CI | 18 | 18 | **17** | **1** | 6 |
| R1 表 4 偏移类型分解 | 24 | 24 | **23** | **1** | 2 |
| R1 §4.7 自建复现 | 20 | 20 | 20 | 0 | 3 |
| R1 §4.7 收敛族敏感性 | 12 | 12 | 11 | **1** | 2 |
| R1 §4.7a 跨架构 | 18 | 18 | 18 | 0 | 2 |
| R1 §4.7b/c | 26 | 26 | 26 | 0 | 0 |
| R1 §4.6-self19 | 12 | 12 | 12 | 0 | 0 |
| R1 附录 A | 6 | 6 | **5** | **1** | 1 |
| R1 其他正文/限制 | 16 | 16 | **15** | **1** | 2 |
| 测量 §4 稳定性地图 | 27 | 27 | **25** | **2** | 3 |
| 测量 §5.2 退化扫描 | 12 | 12 | **10** | **2** | 3 |
| 测量 §5.3 配方扫描 | 24 | 24 | **23** | **1** | 2 |
| 测量 §5.4 AUC + 结论 | 9 | 9 | 9 | 0 | 1 |
| **合计** | **306** | **306** | **297** | **9** | **35** |

---

### 问题 1【阻断投稿】R1 表 4 把 Pearson 系数标为 Spearman ρ

- **位置**：`paper/main.tex:302`（表头）、`:292`、`:319`、`:321`
- **根因**：`code/analyze_corruption_dual.py:119-122` 用 `np.corrcoef`（**Pearson**），脚本自身表头却写「pooled ρ」（`:131`），论文沿用该错误标注。论文正文明确声明该表是 Spearman。
- **实测差异**（独立重算 `data/official/df_fig15-20.pkl`，n=43）：

| 场景 | 论文值（标 ρ） | 实为 Pearson | 真 Spearman | 后果 |
|---|---|---|---|---|
| fog / brightness / contrast | +0.987 / +0.994 / +0.965 | 一致 | 仅舍入 | 无 |
| gaussian_noise | ID −0.087，秩 8/12 | 一致 | −0.109，秩 **6/12** | 秩不同 |
| pixelate | ID +0.215，秩 6/12 | 一致 | +0.191，秩 **2/12** | 秩不同 |
| jpeg_compression | sparsity +0.602，ID 秩 11/12 | +0.6015 | **utility +0.262 为最佳**，ID 秩 **7/12** | **最佳标记身份与排名均改变** |

- **复现命令**：
  ```bash
  python - <<'EOF'
  import numpy as np, pandas as pd
  from scipy.stats import spearmanr
  d=pd.read_pickle('data/official/df_fig15-20.pkl')['jpeg_compression']
  acc=d['acc_test'].to_numpy(float); y=d['acc_test_c'].to_numpy(float); g=d['arch_name'].to_numpy()
  keep=np.abs(acc-acc.mean())<=2.0
  def gres(v,gg):
      v=np.asarray(v,float); o=v.copy()
      for k in set(gg):
          m=np.array([x==k for x in gg]); o[m]-=v[m].mean()
      return o
  for mk in ['sparsity_id','utility_id','acc_test']:
      x=d[mk].to_numpy(float)
      print(mk,'residPearson',round(float(np.corrcoef(gres(x[keep],g[keep]),gres(y[keep],g[keep]))[0,1]),4),
                'residSpearman',round(float(spearmanr(gres(x[keep],g[keep]),gres(y[keep],g[keep]))[0]),4))
  EOF
  ```
  实测：`sparsity_id` residSpearman +0.2160；`utility_id` residSpearman +0.2619；`acc_test` residSpearman −0.0550
- **修法（二选一，推荐 A）**：
  - **A（改脚本对齐论文）**：`code/analyze_corruption_dual.py:119-122` 改用 `spearmanr(...)[0]`，字段名改为 `pooled_rho`/`resid_rho`，重跑更新 `results/corruption_dual_protocol.json`；论文表 4 六行数值与三处秩按新值更新；`:292`/`:319`/`:321` 措辞同步。**注意 pixelate/jpeg 的"最佳标记 = sparsity"结论会改变，`:319` 的定性论断需重写。**
  - **B（改论文标注）**：表头 `$\rho$` → `$r$`，`$\rho_{\text{resid}}$` → `$r_{\text{resid}}$`，`:292` 改为 "pooled Pearson $r$"；同时改脚本 `:131` 表头避免再次误导。`:321` 的 "$-0.12$ (pooled) to $+0.22$ (Protocol A)" 在 Pearson 口径下正确，保留。

### 问题 2【需修改】表 3 的"稳健率 0%"实为 P(r>0)，真实稳健率 82.9%

- **位置**：`paper/main.tex:280`（表 3 行）、`:285`（正文）
- **事实**：`results/ind48_robust_results.json` 的 `cohorts.converged36.bootstrap.id_acc.CIFAR-100` 只有 `p_positive = 0.0`，**没有 `robust_rate` 字段**。论文把 `P(r>0)=0%` 直接填进了 "Bootstrap robust rate" 列。
- **按 `code/bootstrap_ci.py:130-132` 定义重算**（robust = `sign(r)==sign(rho)` 且 `|rho|>=0.5`）：
  ```
  converged36  numrank  CIFAR-100  r_ci=[+0.8191,+0.9425]  robust_rate=1.0000
  converged36  IDacc    CIFAR-100  r_ci=[-0.8026,-0.4195]  robust_rate=0.8290
  ```
  即真实稳健率是 **83%**，不是 0%。full48 对照：numrank 100%（论文 ✓）、IDacc 25.1%（论文写 23%，归档 `bootstrap_ci_results.json` 为 0.23，因 seed 不同略有差异）。
- **修法**：`paper/main.tex:280` 的 `0\%` → `83\%`；`:285` 的 "and 0\% over the converged 36" → "and 83\% over the converged 36"。更根本的做法是让 `code/analyze_ind48_robust.py` 在 `bootstrap` 字典中输出 `robust_rate`（复用 `bootstrap_ci.py` 判据），并在 `audit_claims.py` 中新增对应断言。

### 问题 3【需修改】模型数 78 / 79 / 80 三种说法并存

- **独立复核**（`data/self_trained/ind_validation_results.json` + `mod_arch_results.json`）：
  ```
  trained: 80  {resnet18: 48, regnet_y_400mf: 16, shufflenet_v2_x1_0: 16}
  healthy: 79  {resnet18: 48, regnet_y_400mf: 16, shufflenet_v2_x1_0: 15}
  diverged: shufflenet_v2_x1_0_lr0.5_wd0.0002_s1  (id_acc=0.1)
  ```
  → 恰好 **1 个**发散，不是每个架构各 1 个。
- **论文中 11 处需统一**：`main.tex:59`（摘要）、`:73`、`:92`、`:169`、`:185`、`:392`（✓79）、`:406`（✓79/80）、`:511`、`:524`、`:526`、`:531`（✓80）
- **具体冲突**：
  - `:169` 写 "78 self-trained models --- 48 ResNet-18 ... plus 16 ShuffleNetV2 and 16 RegNetY" → 48+16+16=80，与同句矛盾
  - `:511` 写 "(one lr-seed cell each diverged)" → 实际只有 1 个
  - `:526` 写 "the 78 trained checkpoints" vs `:531` 写 "the 80 trained checkpoints" → 直接矛盾（README 与 DATA_AVAILABILITY.md 均写 80）
- **修法**：统一为「**80 trained / 79 healthy**」双口径，并在首次出现处定义；`:511` 改为 "(one lr-seed cell diverged and was excluded)"。

### 问题 4【需修改】附录 A 的 pooled r=+0.48 实来自 224-model release

- **位置**：`paper/main.tex:583`
- **原文**：该段声称所有数值都算自 80-model release，"pooled $r=+0.48$ is the raw ... correlation in the same release"
- **独立复核**：
  ```
  80-model release  pooled r : +0.4426   ← 论文所在的 release
  224-model release pooled r : +0.4759   ← +0.48 实际来自这里
  ```
  同段的残差化 Spearman 部分正确（D_eff +0.81、Ψ +0.80、ID −0.61、PR +0.79 ✓）；但 `mean_dist ρ=+0.68` 只有 `mean_distance_avg_id`（avg 变体）符合，`mean_distance_id` 是 +0.6160，论文未标明是 avg 变体。
- **修法**：`:583` 的 "+0.48" → "+0.44"；补一句说明 §4.2/§5 中的 +0.48 来自 224-model release；"mean\_dist" → "mean\_dist (avg)"。

### 问题 5【需修改】测量论文退化扫描表的四个数值无原始数据支撑，审计检查恒真

- **位置**：`paper/measurement/main.tex:349-365`（表 3）、`code/analyze_degeneration.py:55-59`、`code/audit_paper_measurement.py:197-204`
- **事实**：
  - 表中 16 个数（316/287/128/271；0.934/0.910/0.586/0.899；0.531/0.438/0.264/0.309）**在仓库中没有任何原始数据文件**。`code/analyze_degeneration.py:55-59` 把它们写成 Python 字面量常量，来源是 `code/archive/smoke_geometry.py` 冒烟测试（用 ResNet-50 + 人脸/物体数据集，与论文所述 CIFAR-10 ResNet-18 训练网络不一致）。
  - `code/audit_paper_measurement.py:198-204` 的检查**恒真**：先定义 `DEG = {"clean": (316, 0.531), ...}`，再 `check("degen", "clean dim", 316, DEG["clean"][0], tol=0)` —— 拿字面量跟自己比，永远通过。
  - 相对变化 59% / 50% 独立验证自洽（(316−128)/316=59.5%，(0.531−0.264)/0.531=50.3%）。
- **修法**：① 补入原始输出（如 `results/degradation_sweep_raw.json`）替换字面量常量；② `audit_paper_measurement.py` 改为从该 JSON 读取重算，消除恒真检查；③ 若无法补数据，在 Limitations 中明确标注该表数值来源为未发布的冒烟测试。

### 问题 6【建议】R1 §3.3 的稳健率 54% 应为 55%

- **位置**：`paper/main.tex:154`
- **事实**：`results/bootstrap_ci_results.json` → `B1_fig3_224.participation_ratio_id.robust_rate = 0.5535` = 55.35%，归档报告 `results/bootstrap_ci_report.md:12` 打印为 `55%`。54% 无法从任何脚本复现。
- **修法**：`54\%` → `55\%`。

### 问题 7【建议】§3.3 的 "several markers" 未逐条断言

- **位置**：`paper/main.tex:153`
- **事实**：`center_alignment_id.robust_rate = 0.1965`（<20% ✓ 正确），但"several markers"（复数）无逐条保护；`axis_alignment_id` = 14.40%、`center_axis_alignment_id` = 19.15% 同样 <20%。
- **修法**：在 `audit_claims.py` 中补三条断言。

### 问题 8【建议】测量论文表 1 的 `utility` 分类标签与归档 JSON 不符

- **位置**：`paper/measurement/main.tex:256`
- **事实**：论文写 `utility` 分类为 "magnitude collapse"，但 `results/marker_stability_map.json` 中 `utility` 的 `classification` 是 **"consistent sign, variable magnitude"**（与 numerical rank 相同）。标签是作者自造的。
- **修法**：改为 JSON 的分类，或在表 1 caption 中说明这是作者的定性解读而非脚本分类。

### 问题 9【建议】RegNetY 架构内样本 n=12 实为 13

- **位置**：`paper/main.tex:511`
- **事实**：按 `audit_claims.py` 的 `within_arch()` 匹配规则：RegNetY n=**13**、ShuffleNetV2 n=14、ResNet18 n=46。
- **修法**：`$n=12$` → `$n=13$`。

### 问题 10【建议】测量论文 "best 与 second-best 相差 0.003" 实为 0.001

- **位置**：`paper/measurement/main.tex:433`、`code/audit_paper_measurement.py:192`
- **事实**：`results/c2_results_seeds0-1-2.json` 中 best 三种子均值 = 0.864333、aug_only = 0.865333，差 **0.001**。且 aug_only 均值**高于** best，故 "best" 严格说不成立。审计脚本容差 0.006 恰好掩盖了偏差。
- **修法**：论文 `0.003` → `0.001`；脚本容差收紧到 0.0005。

---

## 第三部分：审计盲区（CI 当前保护不到）

以下断言论文写了但 `audit_claims.py` / `audit_paper_measurement.py` 未覆盖。**35 处**。

### R1（`code/audit_claims.py`）

| # | 位置 | 断言 | 建议 |
|---|---|---|---|
| B1 | `:59` | 「78 models we trained」 | 加 trained=80 / healthy=79 断言 |
| B2 | `:185` | 「78 healthy models」 | 加 healthy=79 断言 |
| B3 | `:408` | 「OOD span 100% / 93%, ID span 3.7pp」 | 加 3 条断言 |
| B4 | `:228` | 「D_eff 在五个架构全为正 (+0.54~+0.97)」 | 加 5 条逐架构断言 |
| B5 | `:240` | 「80-model baseline 在 b=2.5 达 −0.61」 | 加 1 条 |
| B6 | `:285` | 「robust rate 23% / 0%」 | 见问题 2，先让脚本输出 `robust_rate` |
| B7 | `:129` | 「b=8pp 保留 223/224 和 80/80」 | 80/80 未编码 |
| B8 | `:167` | 「OOD 列在 6 场景间完全相同」 | **当前是硬编码 `"pass": True` 的假断言**（`audit_claims.py:463-467`），应改为真实计算 15 对场景的交叉验证 |
| B9 | `:583` | 附录 A 的 5 个残差化 Spearman 值 | 见问题 4 |
| B10 | `:503` 附近 | 19 个腐蚀场景的 ID-acc 逐场景秩 | 现仅 5 个场景有保护 |
| B11 | `:481` | 「高 ID 的 VGG/MobileNet 迁移更差」 | 定性，可加秩相关断言 |

### 测量论文（`code/audit_paper_measurement.py`）

| # | 位置 | 断言 | 建议 |
|---|---|---|---|
| B12 | `:36` | 「seventeen markers」 | 加 marker count=17 断言 |
| B13 | `:40` | 「Four are stable across every cell」 | 加 robust class count=4 断言 |
| B14 | `:256` | `utility` 分类为 "magnitude collapse" | 见问题 8 |
| B15 | `:262` | `NC1` 分类为 "sign flip (Spearman)" | 与 JSON 的 `SIGN FLIPS (Spearman)` 对齐 |
| B16 | `:401` | 表 4 的 p 值（0.0012/0.0027/0.0070/0.1678） | 现只检查 r 与 rho |
| B17 | `:433` | best/second-best 差 0.003 | 见问题 10，收紧容差 |
| B18 | `:444` | AUC 0.866 / 0.750 / 0.606 | 三个 AUC 均未编码；计算逻辑在 `code/archive/` 下，CI 不跑 |
| B19 | `:434` | 「effective dimension 对九个网络取同一值」 | 加 `nunique()==1` 断言 |
| B20 | `:349` | 退化扫描四个条件 | 见问题 5 |

**特别提示**：B8 是当前审计体系里最危险的一条 —— `audit_claims.py:463-467` 用硬编码的 `"pass": True` 声称"已验证 OOD 列在 6 场景间完全相同"。虽然该结论经独立重算确实成立（15 对场景组合全部 corr=1.0000、maxdiff=0），但**这条"保护"是假的**，它不验证任何东西。

---

## 第四部分：已核对且确认一致（306 项中的 297 项）

**R1**：§4.2 全样本景观 4 项；表 2 全部 7 行 × 3 列 + n=106 + ID span 1.92pp + OOD span 91.8% + 全部 p 值；表 2 baseline 逐架构符号 +0.48/−0.48/+0.46/+0.24/+0.42；§3.3 全部 4 项条件承重论证（含 center_alignment r=+0.4829/ρ=+0.4003、PR resid r=+0.6586、PR 逐架构 +0.93/−0.28/+0.71/+0.94/+0.21、D_eff 逐架构 +0.54~+0.97）；§4.4 全部 7 个带宽点；表 3 其余 9 行 CI；§4.7 表 5 全部 12 项 + 表 6 逐 lr 族 4 行 + 收敛族 12 项；§4.7a 跨架构 18 项；表 8 架构内协议 12 项；§4.7b/c 全部 26 项；§4.6-self 19 场景全部 12 项；80-release 的 OOD 列在 6 场景间逐元素完全相同。

**测量论文**：表 1 全部 9 行 × 3 列 r 值；半径/效用/维度带宽扫描全部轨迹（含旧值 −0.229 已改为实际 −0.246 ✓）；表 4 配方扫描全部 4×3 + 2×2 值（含全部 p 值）；高簇 n=9、local_smoothness r=−0.0534/p=0.8914、effective_dim 簇内恒为 9.0；AUC 四个值；退化表相对变化 59%/50%。

**数据与实验设置**：论文引用的全部数据文件存在（`data/official/df_fig3.pkl`、`df_fig15-20.pkl`、`data/self_trained/{ind_validation_results,mod_arch_results,glue_own48_results}.json`）；7 张图全部存在且 `make_figures.py` / `make_figures_measurement.py` 可重跑复现（PDF 字节数一致）；自训练超参与 `code/train_grid.py` 一致（200 epoch、cosine、SGD momentum 0.9 nesterov、batch 128、lr∈{0.125,0.25,0.5,1.0}×wd∈{2.5e-5,5e-5,1e-4,2e-4}×seed{0,1,2}）。

**CI 链路确认**：`audit_paper_measurement.py` 由 `audit_claims.py` 末尾以 subprocess 方式间接调用，两者任一 FAIL 都会触发 `grep -q "FAIL: 0"` 失败 → CI 红灯。链路是通的（但只跑 `audit_claims.py` + `make_figures.py`，`audit_paper_measurement.py` 未在 workflow 中单独出现）。

---

## 第五部分：无法重算的项（明确声明，不作判断）

1. **自训练 80 个模型本身**需要 GPU 训练与 CIFAR-10/100、Tiny-ImageNet、CIFAR-10-C 下载。本审计仅从已发布的 per-model 测量 JSON 重算，**未验证这些测量值本身的正确性**。
2. **测量论文退化扫描的原始测量**（需遮挡/噪声注入 + 特征提取）无任何发布数据，无法重算 → 按问题 5 处理。
3. **`results/glue_public_validation.json` 的 3 条记录**对应的原始对照模型未发布，无法独立验证 R1 §6 限制条款 3 中「reproduces the released PR and NC1 exactly, and the released radius within ~10%」这一说法。

---

## 修改优先级清单

| 优先级 | 项 | 动作 |
|---|---|---|
| **P0 阻断** | 问题 1 | 修正表 4 的统计量类型标注（改脚本或改论文，二选一） |
| **P0 阻断** | 问题 5 | 补退化扫描原始数据，移除恒真审计检查 |
| **P1 需改** | 问题 2 | 表 3 的 0% → 83% |
| **P1 需改** | 问题 3 | 统一模型数口径为 80 trained / 79 healthy（11 处） |
| **P1 需改** | 问题 4 | 附录 A 的 +0.48 → +0.44，标明 mean_dist (avg) |
| **P1 需改** | §1.3 | 修复 `references.bib` 的 6 处截断 + 2 处 booktitle |
| **P2 建议** | 问题 6-10 | 54%→55%、补三条稳健率断言、utility 分类标签、n=12→13、0.003→0.001 |
| **P2 建议** | 盲区 B8 | 把硬编码的 `"pass": True` 改成真实计算 |
| **P3 建议** | 其余盲区 | 按第三部分表格逐条补入 CLAIMS/CHECKS |
