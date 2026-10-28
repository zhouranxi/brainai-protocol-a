# 投稿时间线决策（2026-10-27 定案）

## 结论：主投 TMLR，arXiv 预印本先行，Zenodo 同步发布

## 为什么改目标（原定 ICLR）

原计划投 ICLR。核对官方截止日期后发现 **ICLR 2027 的投稿截止是 2026-09-25，已过去一个月**。

| 会议 | 2026/27 截止 | 状态（2026-10-27） |
|---|---|---|
| ICLR 2027 | 2026-09-25 | ❌ 已过 1 个月 |
| NeurIPS 2026 | 2026-05-06 | ❌ 已过 |
| EMNLP 2026 | ARR 2026-05-25 | ❌ 已过（但 ARR 滚动） |
| **TMLR** | **无截止，滚动投稿** | ✅ **可用** |

## TMLR 为什么合适

逐条对照我们的论文形态：

| TMLR 特性 | 对我们的价值 |
|---|---|
| **滚动投稿，无截止日期** | 不必赶死线；等原作者回信确认协议后再投也不迟 |
| **arXiv 预印本不破双盲**（FAQ 明确） | 我们打算先发 arXiv 拿 ID —— TMLR 完全允许 |
| **≤12 页走 2 周快审**，>12 页走 4 周 | 当前 18 页（含附录），主文约 13 页 + 附录 |
| **约 9 周出决定**（TMLR 2025 年报中位 91 天） | 比顶会 126 天快 |
| **无篇幅硬限** | 10 条 Limitations、8 张表、完整证据矩阵全部保留 |
| **开放评审、评审公开可见** | 与"可审计"主题气质一致 |
| **明确不偏好方法论/理论** | 我们的贡献正是协议与方法论 |

**风险**：TMLR 在国内认可度不如顶会，且有讨论的"声望差距"。但论文当前处于"证据完备、结论稳固、未经外部审读"的状态——**先让同行审读比抢截稿日更重要**。等一轮 review 拿到意见后，若需要冲击顶会，可在 ICLR 2028（2027年9月截止）用改进版再投。

## 备选路径（若你更看重会议声望）

**ACL 2027**：接受 2027年1月 的 ARR 周期提交。当前 2026-10-27，准备期约 2.5 个月。
- ARR 周期：2026年10月周期（10-12 提交）→ 可 commit 到 NAACL 2027 / COLING 2027（commit 期限 12-23）
- 下一轮 ARR：2027年1月 → ACL 2027
- 注意：ARR 从 2026年10月起**所有作者必须注册 OpenReview 账号 + ORCID**
- EMNLP 2026 明确打击"引用幻觉"和"完全 AI 生成"的论文 —— 我们的引用核查与可复现包正好是加分项

**建议**：主投 TMLR（现在就投，快速拿到同行反馈），同时用 arXiv 锁定版本。若 2027 年 1 月前想投会议，用 ARR 走 NAACL 2027 或 COLING 2027。

## 执行顺序

```
① [已完成] 论文 → TMLR LaTeX（main.tex + references.bib + main.pdf，18 页，编译 0 error）
② [已完成] 可复现发布包（182 文件 / 2.4G / 80 checkpoint / 8 脚本包内自检全通过）
③ [已完成] 致原作者核实函（已发出，不依赖回复）
④ [待办] 提交 arXiv 预印本（cs.LG）→ 拿 arXiv ID
⑤ [待办] Zenodo 发布（concept DOI，version 1.0）→ 拿 Zenodo DOI
⑥ [待办] 在 main.tex 里填入 arXiv ID + Zenodo DOI（camera-ready 时）
⑦ [待办] 提交 TMLR（OpenReview，双盲版本即当前 main.pdf）
⑧ [待办] 3 个工作日无回复 → 给原作者 follow-up
```

## TMLR 特有的准备事项

- **双盲**：当前 `main.tex` 用 `\usepackage{tmlr}`（无 accepted/preprint 选项）→ 自动隐藏作者。camera-ready 时才改 `\usepackage[accepted]{tmlr}` 并填真实作者信息
- **arXiv 预印本**：TMLR 允许。但预印本里应改用 `\usepackage[preprint]{tmlr}`（该选项会去掉 TMLR 标识）
- **LLM 使用披露**：TMLR FAQ 明确要求作者在投稿首页脚注说明使用了 LLM。**这条必须加** —— 我们用了 WorkBuddy 做文献转换与代码辅助
- **主文页数**：当前 18 页含附录。TMLR 以 12 页为快审分界（超了走 4 周），不是硬性上限
- **代码/数据上传**：TMLR 接受 camera-ready 时关联代码与视频 —— 正好对应我们的 Zenodo 记录

## 待确认

- 是否要在 arXiv 预印本里保留"the original study"这类明确指代？ICLR 要求双盲期间引用自己相关工作用第三人称，TMLR 无此严格要求，但预印本**本来就不是双盲**，可以点名。为安全起见，预印本用 `[preprint]` 模式且不匿名化作者。

## 相关文件

- 论文源：`brainai-smoke/paper/tmlr/`（main.tex / references.bib / main.pdf / figures/）
- 发布包：服务器 `/mnt/mydisk/xinguan-brainai/release/brainai-protocol-a`
- 数字审计：`brainai-smoke/results/claims_audit_report.md`（111 条断言全通过）
- 核实函：`brainai-smoke/protocol_verification_email.md`（已发出）
- 决策记录：`brainai-smoke/adr_0015.md`
