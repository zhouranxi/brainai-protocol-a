# 协议核实询问邮件（致原作者）

> **收件人（主）**：`cchou@flatironinstitute.org` —— Chi-Ning Chou
> **抄送（CC）**：`schung@flatironinstitute.org` —— SueYeon Chung
> **发件人**：周燃犀 / Ranxi Zhou `<zhouranxi@foxmail.com>`
> **机构**：湖南中医药大学（Hunan University of Chinese Medicine），计算机科学与技术专业
> **主题**：`Three protocol questions on the two released datasets (df_fig3.pkl / df_fig15-20.pkl)`
>
> **发送状态**：已通过 Agent Mail 发出（发件地址 qsph6648@agent.qq.com），同时保留本文件
> 供你用 foxmail 学术邮箱转发一份。**不依赖回复**——Limitations 第 9 条已写明
> 「协议一致性未获作者确认」这一残余不确定性，主论点不依赖跨 release 比较。

---

## 英文正文（直接复制发送）

```text
Dear Chi-Ning Chou and Prof. Chung,

I am writing with three specific questions about the two released datasets in
the ood-generalization-geometry repository. I am conducting a controlled
re-analysis of the comparison between representational-geometry markers and
in-distribution (ID) test accuracy as out-of-distribution (OOD) predictors, and
my protocol compares measurements across both released files. Before I report
that comparison, I would rather ask than assume.

For context, the protocol restricts to models with near-identical ID accuracy,
residualizes out architecture main effects, reports Pearson and Spearman
coefficients jointly, and requires sign consistency across architectures. On
the two released files, the ID-accuracy baseline comes out with opposite signs
under this protocol (+0.39 on df_fig3.pkl, -0.63 on df_fig15-20.pkl, with
non-overlapping bootstrap intervals). Before interpreting that as a property of
the baseline rather than of the measurement, I have three questions.

QUESTION 1 — Is acc_lp_ood produced by the same probe pipeline in both files?

Both files contain a column acc_lp_ood, and I read it as the CIFAR-100
linear-probe accuracy (frozen feature extractor, new 100-class head trained on
CIFAR-100 train, evaluated on CIFAR-100 test). For df_fig15-20.pkl this column
is identical across all six corruption scenarios, which is consistent with a
scenario-independent class-level transfer measure.

Were the probes for these two files produced with the same configuration — the
same feature layer, the same regularization strength, the same optimisation
budget and stopping criterion? If any of these differed, some of the sign
difference I observe could be attributable to the probe rather than to the
models, and I would want to report the comparison differently.

QUESTION 2 — What are the probe hyperparameters for the released accuracies?

The released files contain only the resulting accuracy values, so the probe
configuration cannot be reconstructed from them. Could you specify, for each of
df_fig3.pkl and df_fig15-20.pkl: the layer the features are taken from, the
probe optimiser and learning rate, the weight decay / regularisation, the
number of epochs or the stopping rule, and the train/validation split used?

QUESTION 3 — Do the two files describe disjoint model populations, and is the
per-architecture hyperparameter grid different between them?

I find that the (architecture, learning rate, weight decay, seed) keys of the
two files do not intersect at all, and that within df_fig3.pkl each
architecture appears to have been trained on its own 4 learning rates x 4
weight decays with 2-3 seeds (48 models per architecture, 32 for
EfficientNetB0), with the union of hyperparameter values spanning 13 learning
rates and 9 weight decays. Is that description accurate, and was the
hyperparameter grid deliberately structured so that the architectures have
comparable ID accuracy (as the paper describes)?

A NOTE ON HOW I WILL REPORT THIS

Regardless of whether I hear back, I will report in the paper that the OOD
column is the same quantity across the two releases, that the two releases
share no models, and that the identity of the probe pipeline across the two
files is something I could not confirm from the released data. My main claims
do not depend on this cross-release comparison: the sign instability I report
also appears within training families in 48 models I trained myself, and on
those same models the geometric markers and ID accuracy turn out to be nearly
uncorrelated with each other (r = -0.17 to -0.04) while the markers correlate
strongly with class-level transfer (r = +0.90 to +0.95).

I am also grateful for the opportunity to mention that I have requested early
access to the GLUE implementation, since a numerical cross-check against the
reference solver would let me verify the marker values I compute on my own
models.

Any of these details you can share would be appreciated; a pointer to the
relevant script or configuration file would be just as useful as the values
themselves.

Thank you for your time and for this work.

Best regards,

Ranxi Zhou
Assistant Professor, Department of Computer Science and Technology
Hunan University of Chinese Medicine
Changsha, Hunan, China
Email: zhouranxi@foxmail.com
```

---

## 中文对照（审阅用，发送不附）

**请求内容**：三问，均为可核实的具体技术问题，不涉及对论文结论的质疑。

1. **两个 release 的 `acc_lp_ood` 是否同一条 probe 流水线产出**（特征层、正则、优化预算、停止准则）？
   —— 因为我发现在 df_fig15-20.pkl 里该列跨 6 个 corruption 场景完全恒定，
   与"场景无关的类级迁移量"一致；若两次探针配置不同，符号差异可能来自测量而非模型。
2. **两个 release 的 probe 超参各是什么**（释放文件只有结果值，无法反推）。
3. **两个文件是否为互斥模型群体、每个架构的超参网格是否不同**
   —— 我核实到键交集为 0，且 df_fig3.pkl 内每架构为 4 lr × 4 wd × 2–3 seed
   （EfficientNetB0 仅 2 seed、32 个模型），全局并集才覆盖 13 lr × 9 wd。

**关键设计（不依赖回复）**：
- 信里主动声明「无论是否收到回复，我都会在论文中写明：OOD 列是同一个量、两次发布无模型重叠、
  probe 流水线一致性无法从发布数据确认」——**作者看到这封信时就知道我们没在钓鱼**，
  反而更可能回复。
- 明确告知主论点不依赖跨 release 比较（自建模型的族内符号翻转 + 正交性证据），
  避免对方误以为我们把全部赌注押在这个对比上。
- 顺带提一句 GLUE early access 申请（在审状态），把两件事连起来。

**发件地址说明**：Agent Mail 的发件地址是 qsph6648@agent.qq.com，与学术邮箱
zhouranxi@foxmail.com 不同。建议收到作者回复后，用 foxmail 邮箱回一封以学术身份确认；
或直接以 foxmail 转发本邮件正文。两种方式都不影响对方回复。

---

## 后续动作（不阻塞论文推进）

| 时间 | 动作 |
|---|---|
| 立即 | 已发出；论文 Limitations 第 9 条已按"未获确认"写定，**不阻塞投稿** |
| 3 个工作日 | 无回复则同一线程礼貌 follow-up 一句 |
| 1–2 周 | 仍无回复 → 保持 Limitations 现状即可。§4.7c 的正交性证据与 §4.7 的族内符号翻转已独立支撑主论点 |
| 收到回复 | 若确认同协议 → Limitations 第 9 条改为"已由作者确认"并致谢；若确认不同协议 → 立即把 §4.5 的跨 release 对比降为观察，主线改由 §4.7/§4.7c 承担（**这条路径已经预先想好，不需要重做分析**） |
