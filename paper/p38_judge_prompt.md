# P38（ADR-0037）判读指令 —— 权威文件，automation 只指向本文件

> 本文件于 2026-11-27 21:40 写入，早于任何 P38 判读。以本文件为准。

## 背景

Lance 压下全部论文投稿，实验线继续。P37（ADR-0036）判读完毕后，发现 R1（M-c 对 M-a 的早期 top-3 召回增益）在 G1 上从第一批的 +20pp 缩到确认批的 +8pp，而 M-c 自身在两个温度下读数相同——变化只能来自 M-a 在温度 0.7 下的读数、或两批题目不同。本批只处理前者。

学校服务器 xinguan@10.10.52.210（SSH 密钥 ~/.ssh/id_tonelens_xinguan，StrictHostKeyChecking=no，远程 python ~/venvs/dl/bin/python）。项目目录 /mnt/mydisk/xinguan-brainai/。

| 臂 | 命令 | 行数 | 日志 | 结果文件 | done 标志（用通配符找） |
|---|---|---|---|---|---|
| A | `--temp 0.0 --mech a --reps 3 --rep-start 0` | 540 | logs/p38_a.log | runs/v2_p38/p38_a_t00_s0r3.json | /tmp/p38_p38_a_t00_s0r3_done |
| B | `--temp 0.7 --mech a --reps 4 --rep-start 3` | 720 | logs/p38_b.log | runs/v2_p38/p38_a_t07_s3r4.json | /tmp/p38_p38_a_t07_s3r4_done |

2026-11-27 21:33 链式发射，臂 A 完成后自动接臂 B。实测 11.4s/条 → 臂 A 约 1.7h、臂 B 约 2.3h，**预计 2026-11-28 01:10 左右全部完成**。

服务器上有检查脚本 `/tmp/p38_check.py`（用法 `~/venvs/dl/bin/python /tmp/p38_check.py`，它读 JSON 的 `len(rows)`、用通配符找 done 标志）。

**查完成状态的三条铁律**（11-27 19:45 的误报教训）：用 `/tmp/p38_check.py` 或读 JSON 的 `len(rows)`；**不要用 `wc -l` 数 JSON**（单行文件，永远 0 行）；done 标志**用 `ls /tmp/p38_*_done` 通配，不要凭记忆拼文件名**。

注意：ssh 远程命令含 python -c 时引号会被吃掉，复杂逻辑写文件 scp 到 /tmp/ 再执行；scp 与 ssh 分开单条发射。

## 判据（ADR-0037 §五，已冻结）

- **H1**：臂 A 中三次输出逐字相同的格子 ≥ 95%（108/180），否则臂 A 作废、报告后停
- **H2**：行数齐（540 / 720）、无 API error
- **Q1（描述性，无闸门）**：温度 0 下的 R1，按 grade 报告增益、W/L、sign p。**G1 明确不设闸门**（功效 10–38%）
- **Q2（有闸门）**：M-a@0.7（10 次取多数）对 M-a@0，按 grade 配对。任一 grade 的 p<0.05 **且** |差|≥10pp → **T-A**；否则 **T-0**。单 grade 触发即算
- Q3、Q4：描述性

## 步骤

### 1. 检查完成状态

SSH 运行 `~/venvs/dl/bin/python /tmp/p38_check.py`。两臂应分别为 540、720 行，各有 done 标志。连接超时重试一次，仍不通则按「服务器不可达」处理。日志有 Traceback 则报告详情。

### 2. 未完成

报告进度与 ETA，创建一个预计完成后 25 分钟的一次性检查任务（prompt 只需一句：读取并严格执行本文件）。不要每 30 分钟续建。
若服务器不可达：先区分 SSH 闪断还是本机 VPN 隧道掉线（RuiJie SSLVPN 网卡 Disconnected 需 Lance 手动重连）。是 VPN 则报告用户、不再续建。**注意：VPN 掉线不影响服务器上的实验**（setsid nohup 脱钩），不要建议重发。

### 3. 完成

scp 拉取两个 JSON 到 github-repo/results/p38_stateless/，核 md5（本地 `md5sum` 对远端 `md5sum`）。

### 4. 判读

先跑自测确认环境没变：`python github-repo/code/p38_selftest.py` 必须输出 `ALL SELF-TESTS PASS`。
然后运行 `PYTHONIOENCODING=utf-8 python github-repo/code/p38_judge.py`。**判读脚本是发射前冻结并经六场景自测的，不得修改。** 它崩溃或报错就如实报告，不要改脚本让它跑通。

### 5. 写入 ADR

判读结果追加到 github-repo/paper/adr_0037.md 末尾新开「十二、判读」一节。**追加式，不改既有任何一行**。按 grade 分层列表；Q1 的 G1 结果必须带上"60 题功效 10–38%，仅描述"的说明；Q3 的四个格子里每格用的是不同的抽样，差异是抽样散布不是效应，要写明。

### 6. 机制文（github-repo/paper/mechanism/main.tex）

**不改表 2、图 1、§early、任何 rank、任何既有判决、Limitations 既有条目。**
- **Q2 判 T-0**（预期）：不改 main.tex。
- **Q2 判 T-A**：在 Limitations「Output-format sampling noise」条目末尾追加一句，陈述哪个 grade、幅度、方向（"the stateless mechanism's set-level reading at temperature 0.7 moved by X pp at stage N relative to temperature 0, ADR-0037"）。数字全部取自 p38_judge 的输出。
- H1 FAIL：不改 main.tex，只报告。

编译（bash）：

```bash
export PATH="/c/c/tinytex/bin/windows:$PATH"
export TEXMFHOME=/c/c/tinytex/texmf-var
cd github-repo/paper/mechanism
pdftex -fmt=pdflatex -interaction=nonstopmode main.tex > /tmp/tex1.log 2>&1
bibtex main > /tmp/bib.log 2>&1
pdftex -fmt=pdflatex -interaction=nonstopmode main.tex > /tmp/tex2.log 2>&1
pdftex -fmt=pdflatex -interaction=nonstopmode main.tex > /tmp/tex3.log 2>&1
```

要求：/tmp/tex3.log 中 0 个以 `!` 开头的行、0 处 undefined、main.log 中 0 处 Overfull。用 PyMuPDF 量行坐标核对无溢出（letter 版心右边界 x1≈540.3，阈值 541.5，超出的行数应为 0）。不要靠看渲染图下结论。

然后运行 `PYTHONIOENCODING=utf-8 python github-repo/code/p35_paper_audit.py`，要求 0 FAIL。若因新增句子导致审计的某条边界断言失败，那是审计脚本需要同步调整的边界情形——调整后必须重跑确认 0 FAIL，并在 commit message 里写明改了哪条边界。**若 main.tex 没有改动则不需要编译。**

### 7. 提交

git add + commit，commit message 写明各闸门判定与关键数字。push 失败则稍后重试一次，仍失败如实报告。

### 8. 报告

向用户完整报告：H1/H2；Q1 表（按 grade：M-c、M-a、增益、W/L、p）并附"G1 仅描述"的说明；Q2 判定与分层明细；Q3 的 2×2；Q4 的直方图（不稳定格数、有偏/公平硬币各多少）；机制文改了什么（或没改）；以及一句对"R1 在 G1 缩水"这个问题的回答——本批能把它的多少归因于温度对 M-a 的影响、多少留给题目差异。

### 9. 日志

在工作区 .workbuddy/memory/ 当日日志（2026-11-28.md，不存在则创建）**追加**判读记录。**不要直接改 MEMORY.md**（长期记忆只记结论与约定，由主会话蒸馏）。

## 纪律

- 中途读数不作判读，只有全量算数。
- 任何分组差异先按 grade 分层再定性。
- 判读脚本不得改。
- 完成状态的检查方法见开头三条铁律。
